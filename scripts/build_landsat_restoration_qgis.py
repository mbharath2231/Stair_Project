#!/usr/bin/env python3
"""Build an acquisition-by-acquisition Landsat restoration and curve review."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    Qgis, QgsApplication, QgsDateTimeRange, QgsFillSymbol, QgsLayoutExporter, QgsLayoutItemPicture,
    QgsLayoutItemShape, QgsLayoutPoint, QgsLayoutSize, QgsPalettedRasterRenderer, QgsProject, QgsReferencedRectangle,
)

from build_ndvi_timeseries_qgis import add_raster, color_legend, date_range, label, layout_page, map_panel
from build_spring_review_qgis import VIRIDIS, ndvi_renderer

SOURCE_CLASSES = (
    (0, "#e0e4e9", "Unresolved NDVI"), (1, "#315b9a", "Observed Landsat"),
    (2, "#f5b547", "Temporal fallback"), (3, "#b75729", "Global correction"),
    (4, "#187963", "Adaptive correction"),
)


def source_renderer(layer):
    return QgsPalettedRasterRenderer(layer.dataProvider(), 1, [
        QgsPalettedRasterRenderer.Class(value, QColor(color), text)
        for value, color, text in SOURCE_CLASSES
    ])


def acquisition_sheet(project, report, rows, layers, extent, stage, dates, name, columns):
    """Place actual acquisition maps in chronological order on a shared scale."""
    title = "Original Landsat" if stage == "original" else "Restored Landsat"
    width = 16 + columns * 98
    height = 68 + math.ceil(len(dates) / columns) * 78
    layout = layout_page(project, name, width, height)
    label(layout, f"Clayton study area | {title} NDVI, {report['year']}",
          8, 5, width - 16, 20, bold=True, height=12)
    label(layout, f"{dates[0]} to {dates[-1]} | Actual Landsat acquisition dates | Shared NDVI scale and map extent",
          8, 20, width - 16, 11)
    for index, day in enumerate(dates):
        x = 8 + 98 * (index % columns)
        y = 33 + 78 * (index // columns)
        label(layout, day, x, y, 92, 12, bold=True, center=True)
        panel = map_panel(layout, project, layers[day, stage], extent, x + 15, y + 9, 62)
        panel.setId(f"{stage}|{day}")
        available = float(rows[day][f"{stage}_valid_fraction"])
        label(layout, f"Available area {available:.1%}", x, y + 71, 92, 9, height=6, center=True)
    note = "5.01 km square at 30 m | QA-clear Landsat observations | Gray indicates missing NDVI"
    if stage == "adaptive":
        note = "5.01 km square at 30 m | Clear pixels retained | Source flags identify temporal fallbacks | Gray indicates missing NDVI"
    color_legend(layout, report["ndvi_display_limits"], VIRIDIS, height - 23, note=note)
    return layout


def all_acquisition_sheets(project, directory, report, rows, layers, extent):
    previews = []
    dates = report["acquisition_dates"]
    for stage, title, stem in (("original", "Original", "original"), ("adaptive", "Restored", "restored")):
        folder = directory / "map_sheets" / stem
        folder.mkdir(parents=True, exist_ok=True)
        for page, start in enumerate(range(0, len(dates), 12), 1):
            subset = dates[start:start + 12]
            name = f"{title} NDVI | {page:02d} | {subset[0]} to {subset[-1]}"
            layout = acquisition_sheet(project, report, rows, layers, extent, stage, subset, name, 4)
            target = folder / f"{page:02d}_{stem}_{subset[0]}_to_{subset[-1]}.png"
            previews.append((layout, target))
        name = f"{title} NDVI | All {len(dates)} dates"
        layout = acquisition_sheet(project, report, rows, layers, extent, stage, dates, name, 8)
        previews.append((layout, directory / f"{stem}_all_acquisition_maps.png"))
    return previews


def build(directory):
    report = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    with (directory / "all_acquisition_ndvi.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["date"]: row for row in csv.DictReader(stream)}
    year = report["year"]
    project = QgsProject.instance()
    project.clear()
    project.setTitle(f"Clayton study area | Original and restored Landsat, {year}")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    acquisitions = project.layerTreeRoot().addGroup("Acquisition comparison | choose a date and layer")
    layers = {}
    for day in report["acquisition_dates"]:
        group = acquisitions.addGroup(day)
        for stage, text, filename in (
            ("original", "Original NDVI", "original_NDVI.tif"),
            ("adaptive", "Restored NDVI", "adaptive_NDVI.tif"),
            ("source", "Restoration source", "restoration_source.tif"),
        ):
            layer = add_raster(project, group, directory / "acquisitions" / day / filename, text)
            layer.setRenderer(source_renderer(layer) if stage == "source"
                              else ndvi_renderer(layer, report["ndvi_display_limits"]))
            layers[day, stage] = layer
        group.setIsMutuallyExclusive(True, 1)
        group.setExpanded(False)
    active = min(report["acquisition_dates"], key=lambda day: abs((date.fromisoformat(day) - date(year, 7, 15)).days))
    acquisitions.setIsMutuallyExclusive(True, report["acquisition_dates"].index(active))
    acquisitions.children()[report["acquisition_dates"].index(active)].setExpanded(True)
    base = layers[active, "adaptive"]
    extent = base.extent()
    project.setCrs(base.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    base.saveNamedStyle(str(directory / "NDVI_shared.qml"))

    calendar_group = project.layerTreeRoot().addGroup("Calendar comparison | use Temporal Controller")
    ranges = {index: date_range(day) for index, day in enumerate(report["calendar_dates"], 1)}
    for stage, text, filename in (
        ("original", "Original Landsat NDVI", f"original_NDVI_{year}_Mar_Oct.tif"),
        ("adaptive", "Restored Landsat NDVI", f"adaptive_NDVI_{year}_Mar_Oct.tif"),
        ("source", "Restoration source", f"restoration_source_{year}_Mar_Oct.tif"),
    ):
        layer = add_raster(project, calendar_group, directory / "annual" / filename, text)
        layer.setRenderer(source_renderer(layer) if stage == "source"
                          else ndvi_renderer(layer, report["ndvi_display_limits"]))
        if layer.bandCount() != len(ranges):
            raise ValueError(f"Unexpected date count in {filename}")
        temporal = layer.temporalProperties()
        temporal.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
        temporal.setFixedRangePerBand(ranges)
        temporal.setIsActive(True)
    calendar_group.setIsMutuallyExclusive(True, 1)
    calendar_group.setItemVisibilityChecked(False)
    project.timeSettings().setTemporalRange(QgsDateTimeRange(
        ranges[1].begin(), ranges[len(ranges)].end(), True, False))
    project.timeSettings().setTimeStep(1)
    project.timeSettings().setTimeStepUnit(Qgis.TemporalUnit.Days)
    project.timeSettings().setFramesPerSecond(3)

    for name, filename in (
        ("Seasonal curve comparison", "seasonal_curve_comparison.png"),
        ("Original Landsat curve", "original_landsat_curve.png"),
        ("Restored Landsat curve", "restored_landsat_curve.png"),
    ):
        layout = layout_page(project, name, 400, 245)
        picture = QgsLayoutItemPicture(layout)
        layout.addLayoutItem(picture)
        picture.attemptMove(QgsLayoutPoint(3, 3))
        picture.attemptResize(QgsLayoutSize(394, 226))
        picture.setPicturePath(str(directory / filename))
        label(layout, "Available-pixel medians | Clear observations are preserved | View coverage with the NDVI curves",
              8, 232, 384, 9)

    # Choose illustrative dates by original target coverage and date only.
    examples = []
    for anchor, low, high in ((date(year, 6, 1), 0.95, 1.01),
                              (date(year, 7, 15), 0.2, 0.8),
                              (date(year, 8, 1), 0, 1e-9)):
        eligible = [day for day in report["acquisition_dates"]
                    if low <= float(rows[day]["original_valid_fraction"]) < high]
        examples.append(min(eligible or report["acquisition_dates"],
                            key=lambda day: abs((date.fromisoformat(day) - anchor).days)))
    maps = layout_page(project, "Acquisition map examples", 300, 290)
    label(maps, f"Clayton study area | Landsat restoration, {year}", 8, 4, 284, 17, bold=True, height=12)
    label(maps, "Illustrative acquisitions selected by original cloud-free coverage", 8, 17, 284, 10)
    label(maps, "Original Landsat NDVI", 40, 28, 100, 11, bold=True, center=True)
    label(maps, "Restored Landsat NDVI", 177, 28, 110, 11, bold=True, center=True)
    for index, day in enumerate(examples):
        top = 39 + index * 74
        label(maps, day, 8, top + 27, 34, 10, bold=True)
        for x, stage in ((53, "original"), (190, "adaptive")):
            map_panel(maps, project, layers[day, stage], extent, x, top, 61)
            available = float(rows[day][f"{stage}_valid_fraction"])
            label(maps, f"Available area {available:.1%}", x - 15, top + 61, 91, 9,
                  height=7, center=True)
    # The shared color legend is compact enough for the portrait example page.
    label(maps, "NDVI scale", 8, 265, 33, 10, bold=True)
    for index, color in enumerate(VIRIDIS):
        x = 45 + index * 35
        swatch = QgsLayoutItemShape(maps)
        swatch.setShapeType(QgsLayoutItemShape.Rectangle)
        swatch.setSymbol(QgsFillSymbol.createSimple({"color": color, "outline_style": "no"}))
        maps.addLayoutItem(swatch)
        swatch.attemptMove(QgsLayoutPoint(x, 268))
        swatch.attemptResize(QgsLayoutSize(9, 4))
        value = report["ndvi_display_limits"][0] + index * 0.2
        label(maps, "0.0" if abs(value) < 1e-9 else f"{value:.1f}", x + 10, 265, 24, 9)
    label(maps, "Gray indicates missing NDVI | Fully masked targets use flagged temporal interpolation", 8, 278, 284, 9)
    sheets = all_acquisition_sheets(project, directory, report, rows, layers, extent)
    target = directory / "landsat_restoration_time_series.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not save {target}")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 160
    for layout, path in [(maps, directory / "acquisition_map_examples.png"), *sheets]:
        if QgsLayoutExporter(layout).exportToImage(str(path), settings) != QgsLayoutExporter.Success:
            raise RuntimeError(f"Could not render {path.name}")
        print(f"Rendered {path.relative_to(directory)}", flush=True)
    expected = len(project.mapLayers())
    project.clear()
    if (not project.read(str(target)) or len(project.mapLayers()) != expected
            or len(project.layoutManager().layouts()) != 4 + len(sheets)
            or not all(layer.isValid() for layer in project.mapLayers().values())):
        raise RuntimeError("The saved acquisition comparison did not reopen completely")
    for layer in project.mapLayers().values():
        temporal = layer.temporalProperties()
        if temporal.isActive():
            for band, interval in ranges.items():
                if temporal.bandForTemporalRange(layer, interval) != band:
                    raise RuntimeError(f"Temporal calendar mismatch for {layer.name()}, band {band}")
    print(f"{target} ({expected} layers, {4 + len(sheets)} layouts, {len(report['acquisition_dates'])} acquisitions)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
