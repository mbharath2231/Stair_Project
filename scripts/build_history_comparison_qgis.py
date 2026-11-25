#!/usr/bin/env python3
"""Build fixed-scale NIR, red and NDVI comparisons with the Python bundled in QGIS."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsLayoutExporter, QgsProject,
                       QgsRasterLayer, QgsReferencedRectangle, QgsSingleBandPseudoColorRenderer)

from build_cloud_fill_qgis import detail_extent, make_layout
from build_spring_review_qgis import VIRIDIS, ndvi_renderer

VIEWS = {"NDVI": (3, (-0.2, 1.0)), "NIR": (1, (0.0, 0.6)), "Red": (2, (0.0, 0.3))}
PRIMARY = ("truth", "fusion", "six_month", "twenty_year")
HEADINGS = ("Landsat reference", "Existing fusion", "Six-month training", "20-year training")


def build(directory, spatial_pattern=None):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    dates = report["patterns"][spatial_pattern]["dates"] if spatial_pattern else report["dates"]
    full_season = spatial_pattern and "one_season" in report["methods"]
    period = "March–October" if full_season else "March–August"
    project = QgsProject.instance()
    project.setTitle("Clayton Township | Spatial correction" if spatial_pattern else "Clayton Township | Training-history comparison")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    parent = project.layerTreeRoot().addGroup("Select variable, date and model")
    labels = {"truth": "QA-clear Landsat reference", **report["methods"]}
    sensor_comparison = report.get("sensor_comparison")
    nonlinear = report.get("nonlinear_comparison")
    layouts = {"training histories": (PRIMARY, HEADINGS, "comparison")}
    if spatial_pattern:
        labels["masked"] = "Landsat with simulated gap"
        if report.get("recurrent_comparison"):
            fit_names = ("twenty_full_spatial_boosting", "twenty_full_cnn_tuned", "pixel_lstm", "conv_lstm")
            fit_headings = ("Neighborhood boosting", "Tuned CNN", "Pixel LSTM", "ConvLSTM")
        elif report.get("cnn_comparison"):
            fit_names = ("twenty_full", "twenty_full_spatial_boosting", "one_season_cnn", "twenty_full_cnn")
            fit_headings = ("2005–2024 Ridge", "2005–2024 Boosting", "2024 CNN", "2005–2024 CNN")
        elif nonlinear:
            choices = nonlinear["selected"]
            fit_names = ("one_season", choices["one_season"], "twenty_full", choices["twenty_full"])
            fit_headings = ("2024 Ridge", "2024 Nonlinear", "2005–2024 Ridge", "2005–2024 Nonlinear")
        elif "twenty_harmonized" in report["methods"]:
            fit_names = ("one_season", "twenty_full", "twenty_harmonized")
            fit_headings = ("2024 season", "20 years\nNative", "20 years\nHarmonized")
        elif full_season:
            fit_names = ("one_season", "twenty_full")
            fit_headings = ("2024 season", "2005–2024 seasons")
        else:
            fit_names = ("six_month", "twenty_matched", "twenty_full")
            fit_headings = ("Six months", "20 years\nMar–Aug", "20 years\nMar–Oct")
        layouts = {
            "Spatial correction stages": (("truth", "masked", "temporal", "global", "adaptive"),
                ("Landsat reference", "Simulated gap", "Temporal", "Global", "Adaptive"), "stages"),
            "Calibrated spatial correction": (("truth", "adaptive", *fit_names),
                ("Landsat reference", "Adaptive", *fit_headings), "calibration"),
        }
    if sensor_comparison:
        for history, models in sensor_comparison["models"].items():
            name = f"sensor comparison, {sensor_comparison['histories'][history]}"
            layouts[name] = (("truth", *models.values()),
                             ("Landsat reference", *(sensor_comparison["labels"][source] for source in models)),
                             f"{history}_sensors")
    all_supported = [row for row in dates if row["scores"]["NDVI"]["common_pixels"]]
    supported = all_supported
    if spatial_pattern and not full_season:
        supported = [row for row in supported if row["date"][5:7] <= "08"]
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
        for row in dates:
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
        active_index = next(i for i, row in enumerate(dates) if row["date"] == selected[0]["date"])
        view.setIsMutuallyExclusive(True, active_index)
        view.setExpanded(variable == "NDVI")
    parent.setIsMutuallyExclusive(True, 0)
    first = next(iter(layers.values()))
    extent = first.extent()
    gap_extent = detail_extent(first, spatial_pattern) if spatial_pattern == "cloud" else extent
    project.setCrs(first.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, first.crs()))
    if spatial_pattern:
        long, short = report.get("settings", {}).get("primary_comparison", fit_names[:2][::-1])

        def rmse_change(row):
            scores = row["scores"]["NDVI"]["methods"]
            return scores[long]["rmse"] - scores[short]["rmse"]

        examples = [min(supported, key=rmse_change), max(supported, key=rmse_change)]
        subtitle = f"{period} 2025 | Dates with the lowest and highest RMSE difference | Shared NDVI scale"
        note = f"Dates selected by NDVI RMSE difference, {labels[long]} minus {labels[short]}."
        if not full_season:
            autumn = [row for row in all_supported if row["date"][5:7] >= "09"]
            if autumn:
                examples.append(min(autumn, key=rmse_change))
                subtitle = "Largest March–August improvement and setback, then largest autumn improvement | Shared NDVI scale"
        methods, headings, _ = layouts["Calibrated spatial correction"]
        rows = [(row["date"], [layers["NDVI", row["date"], name] for name in methods], gap_extent) for row in examples]
        layout = make_layout(
            project, "NDVI | Selected date comparisons", "Clayton Township | Examples of NDVI performance",
            headings, rows, spatial_pattern,
            subtitle=subtitle, note=note,
            legend=tuple((color, f"{i * 0.2 - 0.2:.2f}") for i, color in enumerate(VIRIDIS)),
        )
        settings = QgsLayoutExporter.ImageExportSettings()
        settings.dpi = 180
        if QgsLayoutExporter(layout).exportToImage(str(directory / f"qgis_{spatial_pattern}_ndvi_examples.png"), settings) != QgsLayoutExporter.Success:
            raise RuntimeError("QGIS could not render the selected date comparisons")
    for variable, (_, bounds) in VIEWS.items():
        low, high = bounds
        levels = [low + i * (high - low) / (len(VIRIDIS) - 1) for i in range(len(VIRIDIS))]
        legend = tuple((color, f"{0.0 if abs(value) < 1e-9 else value:.2f}")
                       for color, value in zip(VIRIDIS, levels))
        units = "NDVI" if variable == "NDVI" else f"{variable} surface reflectance"
        for title, (methods, headings, filename) in layouts.items():
            view_extent = gap_extent if filename == "calibration" else extent
            rows = [(row["date"], [layers[variable, row["date"], name] for name in methods], view_extent)
                    for row in selected]
            subtitle = f"Withheld 2025 Landsat acquisitions | {units} | Shared display scale"
            note = "Dates selected by shared prediction coverage. Each target Landsat acquisition was withheld from prediction."
            if spatial_pattern:
                subtitle = f"{spatial_pattern.capitalize()} holdouts, {period} 2025 | {units} | Shared display scale"
                note = "Dates selected by coverage. Scoring uses withheld pixels only. White marks NoData."
            layout = make_layout(
                project, f"{variable} | {title}", f"Clayton Township | {title.capitalize()}",
                headings, rows, spatial_pattern or "scene", subtitle=subtitle, note=note, legend=legend,
            )
            settings = QgsLayoutExporter.ImageExportSettings()
            settings.dpi = 180
            prefix = f"{spatial_pattern}_" if spatial_pattern else ""
            target = directory / f"qgis_{prefix}{variable.lower()}_{filename}.png"
            if QgsLayoutExporter(layout).exportToImage(str(target), settings) != QgsLayoutExporter.Success:
                raise RuntimeError(f"QGIS could not render {target}")
    path = directory / (f"spatial_history_{spatial_pattern}.qgz" if spatial_pattern else "training_history_comparison.qgz")
    if not project.write(str(path)):
        raise RuntimeError(f"QGIS could not save {path}")
    expected = len(layers)
    project.clear()
    layout_count = len(VIEWS) * len(layouts) + int(spatial_pattern is not None)
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
    parser.add_argument("--spatial-pattern", choices=("cloud", "stripes"))
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve(), args.spatial_pattern)
