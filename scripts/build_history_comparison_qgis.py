#!/usr/bin/env python3
"""Build fixed-scale NIR, red and NDVI comparisons with the Python bundled in QGIS."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsLayoutExporter, QgsProject,
                       QgsRasterLayer, QgsReferencedRectangle, QgsSingleBandPseudoColorRenderer)

from build_cloud_fill_qgis import make_layout
from build_spring_review_qgis import VIRIDIS, ndvi_renderer

VIEWS = {"NDVI": (3, (-0.2, 1.0)), "NIR": (1, (0.0, 0.6)), "Red": (2, (0.0, 0.3))}
PRIMARY = ("truth", "fusion", "six_month", "twenty_year")
HEADINGS = ("Landsat reference", "Existing fusion", "Six-month training", "20-year training")


def build(directory):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    project = QgsProject.instance()
    project.setTitle("Clayton Township | Training-history comparison")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    parent = project.layerTreeRoot().addGroup("Select variable, date and model")
    labels = {"truth": "QA-clear Landsat reference", **report["methods"]}
    sensor_comparison = report.get("sensor_comparison")
    layouts = {"training histories": (PRIMARY, HEADINGS, "comparison")}
    if sensor_comparison:
        for history, models in sensor_comparison["models"].items():
            name = f"sensor comparison, {sensor_comparison['histories'][history]}"
            layouts[name] = (("truth", *models.values()),
                             ("Landsat reference", *(sensor_comparison["labels"][source] for source in models)),
                             f"{history}_sensors")
    supported = [row for row in report["dates"] if row["scores"]["NDVI"]["common_pixels"]]
    selected = []
    for month in sorted({row["date"][:7] for row in supported}):
        if int(month[-2:]) % 2:
            continue
        candidates = [row for row in supported if row["date"].startswith(month)]
        selected.append(max(candidates, key=lambda row: (
            row["scores"]["NDVI"]["common_pixels"], -abs(date.fromisoformat(row["date"]).day - 15))))
    if not selected:
        selected = supported[:4]
    layers = {}
    grid = None
    for variable, (band, bounds) in VIEWS.items():
        view = parent.addGroup(variable)
        for row in report["dates"]:
            day = row["date"]
            count = row["scores"][variable]["common_pixels"]
            group = view.addGroup(f"{day} | {count:,} common pixels")
            for name, relative in row["rasters"].items():
                layer = QgsRasterLayer(str(directory / relative), labels[name])
                if not layer.isValid() or layer.bandCount() != 3:
                    raise ValueError(f"Invalid comparison raster {relative}")
                actual = (layer.crs(), layer.extent(), layer.width(), layer.height())
                if grid is None:
                    grid = actual
                elif grid != actual:
                    raise ValueError(f"Comparison raster uses another grid {relative}")
                layer.setRenderer(ndvi_renderer(layer, bounds, band=band))
                project.addMapLayer(layer, False)
                group.addLayer(layer).setExpanded(False)
                layers[variable, day, name] = layer
            group.setIsMutuallyExclusive(True, 0)
            group.setExpanded(day == selected[0]["date"])
        active_index = next(i for i, row in enumerate(report["dates"]) if row["date"] == selected[0]["date"])
        view.setIsMutuallyExclusive(True, active_index)
        view.setExpanded(variable == "NDVI")
    parent.setIsMutuallyExclusive(True, 0)
    first = next(iter(layers.values()))
    extent = first.extent()
    project.setCrs(first.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, first.crs()))
    for variable, (_, bounds) in VIEWS.items():
        low, high = bounds
        levels = [low + i * (high - low) / (len(VIRIDIS) - 1) for i in range(len(VIRIDIS))]
        legend = tuple((color, f"{0.0 if abs(value) < 1e-9 else value:.2f}")
                       for color, value in zip(VIRIDIS, levels))
        units = "NDVI" if variable == "NDVI" else f"{variable} surface reflectance"
        for title, (methods, headings, filename) in layouts.items():
            rows = [(row["date"], [layers[variable, row["date"], name] for name in methods], extent)
                    for row in selected]
            layout = make_layout(
                project, f"{variable} | {title}", f"Clayton Township | {title.capitalize()}",
                headings, rows, "scene", subtitle=f"Withheld 2025 Landsat acquisitions | {units} | Shared display scale",
                note="Dates selected by shared prediction coverage. Each target Landsat acquisition was withheld from prediction.",
                legend=legend,
            )
            settings = QgsLayoutExporter.ImageExportSettings()
            settings.dpi = 180
            target = directory / f"qgis_{variable.lower()}_{filename}.png"
            if QgsLayoutExporter(layout).exportToImage(str(target), settings) != QgsLayoutExporter.Success:
                raise RuntimeError(f"QGIS could not render {target}")
    path = directory / "training_history_comparison.qgz"
    if not project.write(str(path)):
        raise RuntimeError(f"QGIS could not save {path}")
    expected = len(layers)
    project.clear()
    layout_count = len(VIEWS) * len(layouts)
    if (not project.read(str(path)) or len(project.mapLayers()) != expected
            or len(project.layoutManager().layouts()) != layout_count):
        raise RuntimeError("QGIS could not reopen all comparison layers and layouts")
    for layer in project.mapLayers().values():
        renderer = layer.renderer()
        if not layer.isValid() or not isinstance(renderer, QgsSingleBandPseudoColorRenderer):
            raise RuntimeError("QGIS reopened an invalid layer or renderer")
        _, bounds = next(settings for settings in VIEWS.values() if settings[0] == renderer.band())
        if (renderer.classificationMin(), renderer.classificationMax()) != bounds:
            raise RuntimeError("QGIS changed a shared display scale")
    print(f"{path} ({expected} styled layers, {layout_count} layouts)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
