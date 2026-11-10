#!/usr/bin/env python3
"""Run with QGIS Python to compare the five TIFFs using one display stretch."""
from __future__ import annotations

import argparse
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsColorRampShader, QgsContrastEnhancement,
    QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemMap, QgsLayoutPoint,
    QgsLayoutSize, QgsMultiBandColorRenderer, QgsPrintLayout, QgsProject,
    QgsRasterLayer, QgsRasterShader, QgsRectangle, QgsReferencedRectangle,
    QgsSingleBandPseudoColorRenderer,
)


STAGES = {
    "01_original": "01  Original",
    "02_cloud_masked": "02  Cloud masked",
    "03_temporal": "03  Temporal",
    "04_global": "04  Global",
    "05_adaptive": "05  Adaptive",
}
RAMP = [(0.015, "#440154"), (0.0375, "#3b528b"), (0.060, "#21918c"),
        (0.0825, "#5ec962"), (0.105, "#fde725")]
# Rounded 2nd and 98th percentiles of the QA-clear July 29 pixels.
FCC_STRETCH = ((0.08, 0.60), (0.005, 0.165), (0.01, 0.15))


def red_renderer(layer):
    low, high = RAMP[0][0], RAMP[-1][0]
    ramp = QgsColorRampShader(low, high)
    ramp.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
    ramp.setColorRampItemList([
        QgsColorRampShader.ColorRampItem(value, QColor(color), f"{value:g}")
        for value, color in RAMP
    ])
    shader = QgsRasterShader(low, high)
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 2, shader)
    renderer.setClassificationMin(low)
    renderer.setClassificationMax(high)
    return renderer


def fcc_renderer(layer, stretch=FCC_STRETCH):
    provider = layer.dataProvider()
    # TIFF bands 1/2/3 store Landsat NIR/Red/Green (5/4/3).
    renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)
    setters = (renderer.setRedContrastEnhancement, renderer.setGreenContrastEnhancement,
               renderer.setBlueContrastEnhancement)
    for band, ((low, high), set_enhancement) in enumerate(zip(stretch, setters), 1):
        enhancement = QgsContrastEnhancement(provider.dataType(band))
        enhancement.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
        enhancement.setMinimumValue(low)
        enhancement.setMaximumValue(high)
        set_enhancement(enhancement)
    return renderer


def renderer_matches(renderer, fcc, stretch=FCC_STRETCH):
    if not fcc:
        return isinstance(renderer, QgsSingleBandPseudoColorRenderer) and renderer.band() == 2
    if not isinstance(renderer, QgsMultiBandColorRenderer):
        return False
    channels = (renderer.redBand(), renderer.greenBand(), renderer.blueBand())
    enhancements = (renderer.redContrastEnhancement(), renderer.greenContrastEnhancement(),
                    renderer.blueContrastEnhancement())
    return channels == (1, 2, 3) and all(
        enhancement
        and enhancement.contrastEnhancementAlgorithm() == QgsContrastEnhancement.StretchToMinimumMaximum
        and (enhancement.minimumValue(), enhancement.maximumValue()) == bounds
        for enhancement, bounds in zip(enhancements, stretch)
    )


def comparison_layout(project, layers, extent, detail, name, fcc):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(330, 183))

    def label(text, x, y, width, height, size=10, color=None, bold=False, center=False):
        item = QgsLayoutItemLabel(layout)
        item.setText(text)
        item.setFont(QFont("Arial", size, QFont.Bold if bold else QFont.Normal))
        if center:
            item.setHAlign(Qt.AlignHCenter)
        if color:
            item.setBackgroundEnabled(True)
            item.setBackgroundColor(QColor(color))
            item.setFontColor(QColor("white") if color in (RAMP[0][1], RAMP[1][1]) else QColor("black"))
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(width, height))

    label("Clayton Township   /   29 July 2022   /   STAIR gap filling", 8, 4, 315, 9, 18, bold=True)
    if fcc:
        label("False-color infrared  |  Landsat 5 / 4 / 3", 8, 15, 105, 8, 11, bold=True)
        label("Shared RGB stretch   /   NIR 0.08–0.60   /   Red 0.005–0.165   /   Green 0.01–0.15",
              115, 15, 205, 8, 10)
    else:
        label("Band 2  |  Red surface reflectance", 8, 15, 80, 8, 11, bold=True)
        for i, (value, color) in enumerate(RAMP):
            label(f"{value:g}", 91 + 22 * i, 15, 21, 7, color=color, center=True)
        label("Same display scale on all five stages", 205, 15, 113, 8, 10)
    label("Full chip   /   5.01 km across", 8, 25, 315, 6, 11, bold=True)
    label("Field detail   /   same area in every panel", 8, 100, 315, 6, 11, bold=True)
    for i, layer in enumerate(layers):
        left = 8 + 63 * i
        label(layer.name(), left, 32, 60, 6, 10, bold=True, center=True)
        label(layer.name(), left, 107, 60, 6, 10, bold=True, center=True)
        for top, bounds in [(38, extent), (113, detail)]:
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(left, top))
            panel.attemptResize(QgsLayoutSize(60, 60))
            panel.setCrs(project.crs())
            panel.setLayers([layer])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(bounds)
            panel.setFrameEnabled(True)
    label("White areas in the masked panel are Landsat QA gaps. The TIFFs retain their stored reflectance values.",
          8, 175, 315, 6, 9)
    project.layoutManager().addLayout(layout)
    return layout


def build(directory, fcc=False):
    name = "FCC comparison" if fcc else "Red band comparison"
    file_stem = "stair_fcc_comparison" if fcc else "stair_red_comparison"
    project = QgsProject.instance()
    project.setTitle(f"STAIR Clayton | {name}")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    group = project.layerTreeRoot().addGroup(f"Five stages | {name}")
    layers = []
    for stem, title in STAGES.items():
        layer = QgsRasterLayer(str(directory / f"{stem}.tif"), title)
        if not layer.isValid() or layer.bandCount() != 3:
            raise ValueError(f"Expected the three-band raster {stem}.tif")
        layer.setRenderer(fcc_renderer(layer) if fcc else red_renderer(layer))
        project.addMapLayer(layer, False)
        group.addLayer(layer).setExpanded(False)
        layers.append(layer)
    group.setIsMutuallyExclusive(True, 2)
    extent = layers[0].extent()
    project.setCrs(layers[0].crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    # Fixed window around the west-central natural cloud gap.
    detail = QgsRectangle(extent.xMinimum() + 30 * 10, extent.yMaximum() - 30 * 147,
                          extent.xMinimum() + 30 * 104, extent.yMaximum() - 30 * 53)
    layout = comparison_layout(project, layers, extent, detail, name, fcc)
    target = directory / f"{file_stem}.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not write {target}: {project.error()}")
    exporter = QgsLayoutExporter(layout)
    settings = exporter.ImageExportSettings()
    settings.dpi = 180
    if exporter.exportToImage(str(directory / f"{file_stem}.png"), settings) != QgsLayoutExporter.Success:
        raise RuntimeError("QGIS could not render the comparison layout")
    project.clear()
    if not project.read(str(target)) or len(project.mapLayers()) != 5 or not all(
        layer.isValid() and renderer_matches(layer.renderer(), fcc) for layer in project.mapLayers().values()
    ):
        raise RuntimeError("Saved QGIS project did not reopen with all five styled layers")
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fcc", action="store_true", help="Use Landsat 5/4/3 color infrared")
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve(), args.fcc)
