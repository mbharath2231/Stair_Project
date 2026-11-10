#!/usr/bin/env python3
"""Use QGIS Python to map a spring NDVI review with a matched FCC layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsColorRampShader, QgsFillSymbol, QgsLayoutItemLabel,
    QgsLayoutItemMap, QgsLayoutPoint, QgsLayoutSize, QgsPalLayerSettings,
    QgsPrintLayout, QgsProject, QgsRasterLayer, QgsRasterShader, QgsReferencedRectangle,
    QgsSingleBandPseudoColorRenderer, QgsTextBufferSettings, QgsTextFormat,
    QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)

from build_stair_qgis import fcc_renderer

SPRING_FCC = ((0.10, 0.50), (0.03, 0.25), (0.04, 0.20))
VIRIDIS = ("#440154", "#443983", "#31688e", "#21918c", "#35b779", "#90d743", "#fde725")


def ndvi_renderer(layer, bounds):
    low, high = bounds
    ramp = QgsColorRampShader(low, high)
    ramp.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
    ramp.setColorRampItemList([
        QgsColorRampShader.ColorRampItem(low + i * (high - low) / (len(VIRIDIS) - 1), QColor(color))
        for i, color in enumerate(VIRIDIS)
    ])
    shader = QgsRasterShader(low, high)
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    renderer.setClassificationMin(low)
    renderer.setClassificationMax(high)
    return renderer


def build(directory):
    report = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    project = QgsProject.instance()
    project.setTitle("Clayton Township | Spring 2024")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    patches = QgsVectorLayer(str(directory / "sample_patches.gpkg"), "Illustrative NDVI patches", "ogr")
    if not patches.isValid() or patches.featureCount() != 3:
        raise ValueError("The three sample patches could not be loaded")
    patches.renderer().setSymbol(QgsFillSymbol.createSimple({
        "style": "no", "outline_color": "255,255,255,255", "outline_width": "0.45",
    }))
    text = QgsTextFormat()
    text.setColor(QColor("white"))
    text.setFont(QFont("Arial", 10, QFont.Bold))
    text.setSize(10)
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setColor(QColor("#172133"))
    buffer.setSize(0.65)
    text.setBuffer(buffer)
    labels = QgsPalLayerSettings()
    labels.fieldName = "site"
    labels.setFormat(text)
    patches.setLabeling(QgsVectorLayerSimpleLabeling(labels))
    patches.setLabelsEnabled(True)
    project.addMapLayer(patches)

    group = project.layerTreeRoot().addGroup("Select one date and view")
    layers = {}
    sources = {"Landsat": ("Landsat_truth", "observed_dates"), "Fusion": ("STAIR_fusion", "fused_dates")}
    for source, (fcc_stem, date_key) in sources.items():
        for day in report[date_key]:
            stems = [("NDVI", f"{source}_NDVI")]
            if day in report["benchmark_dates"]:
                stems.append(("FCC", fcc_stem))
            for style, stem in stems:
                layer = QgsRasterLayer(str(directory / f"{stem}_{day}.tif"), f"{day} | {source} {style}")
                if not layer.isValid() or layer.bandCount() != (3 if style == "FCC" else 1):
                    raise ValueError(f"Could not load {layer.source()}")
                layer.setRenderer(fcc_renderer(layer, SPRING_FCC) if style == "FCC"
                                  else ndvi_renderer(layer, report["ndvi_display_limits"]))
                project.addMapLayer(layer, False)
                group.addLayer(layer).setExpanded(False)
                layers[day, source, style] = layer
    active = report["benchmark_dates"][-1], "Fusion", "NDVI"
    group.setIsMutuallyExclusive(True, list(layers).index(active))
    extent = layers[active].extent()
    project.setCrs(layers[active].crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("March to April comparison")
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(328, 180))

    def label(value, x, y, width, size=10, center=False):
        item = QgsLayoutItemLabel(layout)
        item.setText(value)
        item.setFont(QFont("Arial", size, QFont.Bold))
        if center:
            item.setHAlign(Qt.AlignHCenter)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(width, 8))

    label("Clayton Township  |  March to April 2024", 7, 4, 310, size=17)
    label("FCC row   /   Shared NIR 0.10–0.50, Red 0.03–0.25, Green 0.04–0.20", 7, 15, 310)
    columns = [(day, source) for day in report["benchmark_dates"] for source in sources]
    for index, (day, source) in enumerate(columns):
        x = 7 + 79 * index
        label(f"{day}   /   {source}", x, 25, 74, center=True)
        for style, top in (("FCC", 33), ("NDVI", 107)):
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(x + 5, top))
            panel.attemptResize(QgsLayoutSize(64, 64))
            panel.setCrs(project.crs())
            panel.setLayers([patches, layers[day, source, style]])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(extent)
            panel.setFrameEnabled(True)
    label("NDVI row   /   Shared scale 0.05–0.90   /   A–C mark the same 210 m sample patches", 7, 98, 310)
    label("The Landsat dates in this layout were withheld from their respective fusion runs.", 7, 172, 310)
    project.layoutManager().addLayout(layout)
    target = directory / "spring_review.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not write {target}: {project.error()}")
    count = len(layers) + 1
    project.clear()
    if not project.read(str(target)) or len(project.mapLayers()) != count or not all(
        layer.isValid() for layer in project.mapLayers().values()
    ):
        raise RuntimeError(f"Could not reopen all {count} layers in {target}")
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
