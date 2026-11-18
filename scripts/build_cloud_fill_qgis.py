#!/usr/bin/env python3
"""Build a QGIS project from the Clayton cloud-filling validation GeoTIFFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemMap,
    QgsLayoutPoint, QgsLayoutSize, QgsPrintLayout, QgsProject, QgsRasterLayer,
    QgsRectangle, QgsReferencedRectangle, QgsSingleBandPseudoColorRenderer,
)

from build_stair_qgis import fcc_renderer, renderer_matches
from build_spring_review_qgis import VIRIDIS, ndvi_renderer
from build_watershed_qgis import diverging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stair.validation import make_holdout

# Round outward from the date-specific 2nd and 98th percentiles of the three
# QA-clear scenes. One stretch lets reviewers compare all dates and stages.
STRETCH = ((0.12, 0.57), (0.02, 0.25), (0.035, 0.20))
FILE_NAME = "clayton_cloud_fill.qgz"
FCC_SUBTITLE = "False color (NIR / red / green)  |  Shared reflectance stretch 0.12–0.57 / 0.02–0.25 / 0.035–0.20"


def detail_extent(layer, pattern):
    extent = layer.extent()
    dx = extent.width() / layer.width()
    dy = extent.height() / layer.height()
    ys, xs = make_holdout((layer.height(), layer.width()), pattern).nonzero()
    x = extent.xMinimum() + dx * (xs.min() + xs.max() + 1) / 2
    y = extent.yMaximum() - dy * (ys.min() + ys.max() + 1) / 2
    half = max(dx * (xs.max() - xs.min() + 1), dy * (ys.max() - ys.min() + 1)) / 2 + 240
    return QgsRectangle(x - half, y - half, x + half, y + half)


def make_layout(project, name, title, columns, rows, pattern, *,
                subtitle=FCC_SUBTITLE, note=None, legend=(), background=None):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    width = 49 + 56 * len(columns)
    maps_top = 33
    map_size = 53
    row_step = 58
    footer_y = maps_top + row_step * len(rows)
    height = footer_y + (19 if legend else 9)
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(width, height))

    def label(text, x, y, w, h, size, *, bold=False, center=False, fill=None):
        item = QgsLayoutItemLabel(layout)
        item.setText(text)
        item.setFont(QFont("Arial", size, QFont.Bold if bold else QFont.Normal))
        item.setFontColor(QColor("#1d2b36"))
        if center:
            item.setHAlign(Qt.AlignHCenter)
        if fill is not None:
            item.setBackgroundEnabled(True)
            item.setBackgroundColor(QColor(fill))
        item.setVAlign(Qt.AlignVCenter)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(w, h))

    label(title, 7, 4, width - 14, 9, 17, bold=True)
    label(subtitle, 7, 14, width - 14, 7, 10)
    for column, heading in enumerate(columns):
        x = 43 + 56 * column
        label(heading, x, 25, map_size, 6, 10, bold=True, center=True)

    for row, (heading, layers, extent) in enumerate(rows):
        y = maps_top + row * row_step
        label(heading, 5, y + 21, 37, 9, 9, bold=True)
        for column, layer in enumerate(layers):
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(43 + 56 * column, y))
            panel.attemptResize(QgsLayoutSize(map_size, map_size))
            panel.setCrs(project.crs())
            panel.setLayers([layer])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(extent)
            panel.setFrameEnabled(True)
            if background is not None:
                panel.setBackgroundColor(QColor(background))

    mask_name = "simulated cloud" if pattern == "cloud" else "simulated stripes"
    if note is None:
        note = f"White in the masked panel marks the {mask_name}. Original values inside the mask were reserved for validation."
    label(note, 7, footer_y - 3, width - 14, 7, 9)
    for i, (color, value) in enumerate(legend):
        x = 7 + 22 * i
        label("", x, footer_y + 6, 8, 4, 9, fill=color)
        label(value, x + 9, footer_y + 5, 13, 6, 9)
    project.layoutManager().addLayout(layout)
    return layout


def build(directory, pattern):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if pattern not in report["patterns"] or not report.get("stage_tiffs"):
        raise ValueError("Run cloud density validation with --export-stages for the selected pattern")

    is_ndvi = "ndvi_display_limits" in report
    errors = report.get("error_tiffs", {})
    project = QgsProject.instance()
    project.setTitle("Clayton Township | Landsat cloud filling" + (" NDVI" if is_ndvi else ""))
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    parent = project.layerTreeRoot().addGroup("Comparison dates")
    days = report["dates"]
    columns = list(report["stage_tiffs"].values())
    layers_by_day = {}
    errors_by_day = {}
    grid = None
    for index, day in enumerate(days):
        group = parent.addGroup(day)
        layers = []
        stage_names = report["stage_tiffs"] | errors
        for stem, title in stage_names.items():
            path = directory / pattern / day / f"{stem}.tif"
            layer = QgsRasterLayer(str(path), title)
            if not layer.isValid() or layer.bandCount() != (1 if is_ndvi else 3):
                raise ValueError(f"Could not load the expected raster at {path}")
            actual = (layer.crs(), layer.extent(), layer.width(), layer.height())
            if grid is None:
                grid = actual
            elif actual != grid:
                raise ValueError(f"Layer is on a different grid: {path}")
            if stem in errors:
                layer.setRenderer(diverging(layer, report["error_display_limit"]))
            elif is_ndvi:
                layer.setRenderer(ndvi_renderer(layer, report["ndvi_display_limits"]))
            else:
                layer.setRenderer(fcc_renderer(layer, STRETCH))
            project.addMapLayer(layer, False)
            group.addLayer(layer).setExpanded(False)
            layers.append(layer)
        group.setIsMutuallyExclusive(True, 0)
        group.setExpanded(index == 0)
        layers_by_day[day] = layers[:len(columns)]
        errors_by_day[day] = layers[len(columns):]

    parent.setIsMutuallyExclusive(True, 0)
    first = next(iter(layers_by_day.values()))[0]
    extent = first.extent()
    project.setCrs(first.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, first.crs()))
    layout_options = {}
    title = "Clayton Township | Landsat cloud filling"
    if is_ndvi:
        low, high = report["ndvi_display_limits"]
        levels = [low + i * (high - low) / (len(VIRIDIS) - 1) for i in range(len(VIRIDIS))]
        legend = tuple((c, "0.0" if abs(v) < 1e-9 else f"{v:.1f}") for c, v in zip(VIRIDIS, levels))
        title = "Clayton Township | NDVI by cloud-filling stage"
        layout_options = {
            "subtitle": "NDVI = (NIR − red) / (NIR + red)   |   Shared color scale for all dates",
            "legend": legend,
        }
    overview = make_layout(project, "All dates | five stages", title, columns,
                           [(d, layers, extent) for d, layers in layers_by_day.items()], pattern,
                           **layout_options)
    detail = detail_extent(first, pattern)
    for day, layers in layers_by_day.items():
        make_layout(project, f"{day} | {pattern} detail", f"Clayton Township | {day}",
                    columns, [("Full chip", layers, extent), ("Gap detail", layers, detail)],
                    pattern, **layout_options)

    error_layout = None
    if errors:
        limit = report["error_display_limit"]
        error_layout = make_layout(
            project, "All dates | NDVI errors", "NDVI error inside the simulated gap",
            list(errors.values()), [(day, layers, detail) for day, layers in errors_by_day.items()], pattern,
            subtitle="Restored NDVI minus original NDVI   |   One scale for all methods and dates",
            note="Blue is too low, orange is too high, and light gray is outside the scored area.",
            legend=(("#244da3", f"−{limit:.2f}"), ("#f8f8f8", "0.00"), ("#b64425", f"+{limit:.2f}")),
            background="#dce2e8",
        )
    prefix = "ndvi_" if is_ndvi else "" if pattern == "cloud" else f"{pattern}_"
    target = directory / (prefix + FILE_NAME)
    if not project.write(str(target)):
        raise RuntimeError(f"Could not save {target}: {project.error()}")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 180
    previews = {f"{prefix}five_stage_comparison.png": overview}
    if error_layout is not None:
        previews["ndvi_error_comparison.png"] = error_layout
    for name, layout in previews.items():
        if QgsLayoutExporter(layout).exportToImage(str(directory / name), settings) != QgsLayoutExporter.Success:
            raise RuntimeError(f"QGIS could not render {name}")
    count = len(days) * (len(columns) + len(errors))
    layouts = len(days) + 1 + int(error_layout is not None)
    project.clear()
    if (not project.read(str(target)) or len(project.mapLayers()) != count
            or len(project.layoutManager().layouts()) != layouts
            or not all(layer.isValid() for layer in project.mapLayers().values())):
        raise RuntimeError(f"QGIS could not reopen all {count} styled layers and layouts")
    for layer in project.mapLayers().values():
        renderer = layer.renderer()
        if is_ndvi:
            if layer.name() in errors.values():
                limit = report["error_display_limit"]
                bounds = (-limit, limit)
            else:
                bounds = tuple(report["ndvi_display_limits"])
            valid = (isinstance(renderer, QgsSingleBandPseudoColorRenderer)
                     and (renderer.classificationMin(), renderer.classificationMax()) == bounds)
        else:
            valid = renderer_matches(renderer, True, STRETCH)
        if not valid:
            raise RuntimeError(f"QGIS changed the saved display scale for {layer.name()}")
    print(f"{target} ({count} GeoTIFFs, {layouts} layouts)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", choices=("cloud", "stripes"), default="cloud")
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve(), args.pattern)
