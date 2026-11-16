#!/usr/bin/env python3
"""Build a portable QGIS project and layouts for a completed watershed run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsColorRampShader, QgsCoordinateReferenceSystem,
    QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemMap, QgsLayoutPoint,
    QgsLayoutSize, QgsPalettedRasterRenderer, QgsPrintLayout, QgsProject,
    QgsRasterLayer, QgsRasterShader, QgsRectangle, QgsReferencedRectangle,
    QgsSingleBandPseudoColorRenderer, QgsVectorLayer,
)

from build_stair_qgis import fcc_renderer
from build_spring_review_qgis import ndvi_renderer

STAGES = {
    "01_original": "01 Original reflectance",
    "02_QA_masked": "02 QA masked",
    "03_temporal": "03 Temporal fill",
    "04_global": "04 Global correction",
    "05_adaptive": "05 Adaptive correction",
    "06_adaptive_aligned_30m": "06 Adaptive MODIS on 30 m grid",
    "06_anchor_residual": "06 Landsat minus MODIS anchor residual",
    "07_temporal_source": "07 MODIS temporal fill source",
    "07_fusion_30m": "07 Fusion at 30 m",
    "08_NDVI": "08 Fused NDVI",
    "09_fusion_support": "09 Fusion support",
    "10_anchor_ages": "10 Prior and later anchor distances in days",
    "11_Landsat_only": "11 Landsat temporal baseline",
    "12_same_anchor_Landsat": "12 Landsat interpolation using fusion anchors",
    "13_NDVI_error": "13 Fusion minus withheld Landsat NDVI",
}


def diverging(layer, limit):
    ramp = QgsColorRampShader(-limit, limit)
    ramp.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
    ramp.setColorRampItemList([
        QgsColorRampShader.ColorRampItem(value, QColor(color))
        for value, color in [(-limit, "#244da3"), (0, "#f8f8f8"), (limit, "#b64425")]
    ])
    shader = QgsRasterShader(-limit, limit)
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(-limit)
    renderer.setClassificationMax(limit)
    return renderer


def style(layer, stage):
    if stage == "09_fusion_support":
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, [
            QgsPalettedRasterRenderer.Class(0, QColor("#ae3e22"), "Unresolved"),
            QgsPalettedRasterRenderer.Class(1, QColor("#237a63"), "Two-sided fusion"),
        ]))
    elif stage == "07_temporal_source":
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, [
            QgsPalettedRasterRenderer.Class(0, QColor("#ae3e22"), "Unresolved"),
            QgsPalettedRasterRenderer.Class(1, QColor("#237a63"), "QA-approved observation"),
            QgsPalettedRasterRenderer.Class(2, QColor("#34a8d3"), "Two observations within 8 days each"),
            QgsPalettedRasterRenderer.Class(3, QColor("#9056b1"), "At least one observation more than 8 days away"),
        ]))
    elif stage == "08_NDVI":
        layer.setRenderer(ndvi_renderer(layer, (0, 1)))
    elif stage == "13_NDVI_error":
        layer.setRenderer(diverging(layer, .20))
    elif stage == "06_anchor_residual":
        layer.setRenderer(diverging(layer, .20))
    elif layer.bandCount() == 3:
        layer.setRenderer(fcc_renderer(layer))


def make_layout(project, extent, title, rows, columns, notes):
    width = 44 + 62 * len(columns)
    height = 37 + 44 * len(rows)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(title)
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(width, height))

    def label(value, x, y, w, size=10, center=False):
        item = QgsLayoutItemLabel(layout)
        item.setText(value)
        item.setFont(QFont("Arial", size, QFont.Bold))
        if center:
            item.setHAlign(Qt.AlignHCenter)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(w, 8))

    label(title, 5, 3, width - 10, 17)
    label("River Raisin watershed  |  Shared FCC stretch in NIR, red and green", 5, 13, width - 10)
    for index, (column, column_layers) in enumerate(columns):
        x = 43 + 62 * index
        label(column, x, 22, 58, 10, True)
        for row, layer in enumerate(column_layers):
            if layer is None:
                continue
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(x, 30 + 44 * row))
            panel.attemptResize(QgsLayoutSize(58, 41))
            panel.setCrs(project.crs())
            panel.setLayers([layer])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(extent)
            panel.setFrameEnabled(True)
    for row, heading in enumerate(rows):
        label(heading, 4, 46 + 44 * row, 38)
    label(notes, 5, height - 9, width - 10, 9)
    project.layoutManager().addLayout(layout)
    return layout


def add_gap_overlays(directory, project, groups, first_day):
    diagnostics = directory / "gap_diagnostics"
    summary = json.loads((diagnostics / "summary.json").read_text(encoding="utf-8"))
    colors = ["#34b7e9", "#efb634", "#866dee", "#ec5381", "#1558c0", "#f36520", "#282336"]
    reasons = [
        QgsPalettedRasterRenderer.Class(int(code), QColor(color), label)
        for (code, label), color in zip(summary["codes"].items(), colors, strict=True)
    ]
    count = 0
    for day, info in summary["dates"].items():
        group = groups[f"{day}_Prediction"]
        for file_key, title, classes in (
            ("gap_overlay", "Diagnostics | Fusion gap reasons", reasons),
            ("qa_overlay", "Diagnostics | Pixels removed by MODIS QA", [
                QgsPalettedRasterRenderer.Class(1, QColor("#ef10cc"), "Excluded by mandatory QA"),
            ]),
        ):
            layer = QgsRasterLayer(str(diagnostics / info[file_key]), title)
            if not layer.isValid() or layer.bandCount() != 1:
                raise ValueError(f"Could not load {layer.source()}")
            layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
            project.addMapLayer(layer, False)
            node = group.addLayer(layer)
            visible = day == first_day and file_key == "gap_overlay"
            node.setExpanded(visible)
            node.setItemVisibilityChecked(visible)
            count += 1
    return count


def build(directory, *, gap_review=False):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    project = QgsProject.instance()
    project.setTitle("River Raisin watershed | Five-day STAIR fusion")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setCrs(QgsCoordinateReferenceSystem(report["grid"]["crs"]))
    boundary = QgsVectorLayer(str(directory / "watershed.gpkg"), "River Raisin watershed boundary", "ogr")
    if not boundary.isValid():
        raise ValueError("Invalid watershed boundary")
    project.addMapLayer(boundary)
    project.layerTreeRoot().findLayer(boundary.id()).setItemVisibilityChecked(False)

    groups = {}
    for role, dates in (("Prediction days", report["days"]), ("Restored anchor pairs", report["anchors"])):
        parent = project.layerTreeRoot().addGroup(role)
        for day in dates:
            group = f"{day}_{'Prediction' if role == 'Prediction days' else 'Anchor'}"
            suffix = ""
            info = report["dates"][day]
            if "held_out" in info:
                suffix = " | withheld Landsat"
            elif not info["landsat_acquired"]:
                suffix = " | no Landsat acquisition"
            groups[group] = parent.addGroup(day + suffix)

    overlay_count = add_gap_overlays(directory, project, groups, report["days"][0]) if gap_review else 0
    layers = {}
    active = (f'{report["days"][0]}_Prediction', "STAIR", "07_fusion_30m")
    for entry in report["layers"]:
        group, sensor, stage = (entry[k] for k in ("group", "sensor", "stage"))
        name = f"{sensor} | {STAGES[stage]}"
        layer = QgsRasterLayer(str(directory / entry["file"]), name)
        if not layer.isValid() or layer.bandCount() != len(entry["bands"]):
            raise ValueError(f"Could not load {entry['file']}")
        style(layer, stage)
        project.addMapLayer(layer, False)
        node = groups[group].addLayer(layer)
        node.setExpanded(False)
        node.setItemVisibilityChecked((group, sensor, stage) == active)
        layers[group, sensor, stage] = layer
    for group in groups.values():
        group.setExpanded(False)
    active_group = groups[active[0]]
    active_group.setExpanded(True)
    extent = QgsRectangle(layers[active].extent())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))

    daily_columns = []
    for day in report["days"]:
        group = f"{day}_Prediction"
        coverage = report["dates"][day]["fusion"]["coverage"]
        daily_columns.append((f"{day}   |   {coverage:.2%}", [
            layers[group, "MODIS", "02_QA_masked"],
            layers[group, "STAIR", "07_fusion_30m"],
            layers[group, "STAIR", "08_NDVI"],
        ]))
    overview = make_layout(project, extent, "Five-day forecast coverage",
                           ("MODIS QA", "STAIR fusion", "Fused NDVI"), daily_columns,
                           "Percentages show valid three-band fusion inside the watershed. NDVI uses a shared 0 to 1 scale.")

    for day in report["days"]:
        if "held_out" not in report["dates"][day]:
            continue
        group = f"{day}_Prediction"
        columns = [
            (label, [layers[group, "Landsat", stage], layers[group, "MODIS", stage]])
            for label, stage in (
                ("Original", "01_original"), ("QA masked", "02_QA_masked"),
                ("Temporal", "03_temporal"), ("Global", "04_global"),
                ("Adaptive", "05_adaptive"),
            )
        ]
        make_layout(project, extent, f"{day} sensor restoration stages", ("Landsat", "MODIS"), columns,
                    "The Landsat row shows gap filling on the observed scene as a separate diagnostic.")
        make_layout(project, extent, f"{day} withheld Landsat validation", ("Reflectance",), [
            ("Withheld Landsat", [layers[group, "Landsat", "02_QA_masked"]]),
            ("STAIR fusion", [layers[group, "STAIR", "07_fusion_30m"]]),
            ("Landsat temporal", [layers[group, "Validation", "11_Landsat_only"]]),
            ("Same-anchor Landsat", [layers[group, "Validation", "12_same_anchor_Landsat"]]),
            ("Restored MODIS", [layers[group, "MODIS", "06_adaptive_aligned_30m"]]),
        ], "Landsat from all five prediction dates was withheld from fusion.")

    target = directory / ("raisin_fusion_gap_review.qgz" if gap_review else "raisin_five_day_fusion.qgz")
    if not project.write(str(target)):
        raise RuntimeError(f"Could not write {target}: {project.error()}")
    if not gap_review:
        settings = QgsLayoutExporter.ImageExportSettings()
        settings.dpi = 135
        preview = directory / "five_day_coverage.png"
        if QgsLayoutExporter(overview).exportToImage(str(preview), settings) != QgsLayoutExporter.Success:
            raise RuntimeError(f"Could not render {preview}")
    project.clear()
    count = len(layers) + overlay_count
    if (not project.read(str(target)) or len(project.mapLayers()) != count + 1
            or not all(layer.isValid() for layer in project.mapLayers().values())):
        raise RuntimeError(f"Could not reopen the complete project at {target}")
    print(f"{target} ({count} raster layers and watershed boundary)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--gap-review", action="store_true", help="Build a separate project with QA and gap overlays")
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve(), gap_review=args.gap_review)
