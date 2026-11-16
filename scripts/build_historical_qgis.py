#!/usr/bin/env python3
"""Build the 2005 Clayton reflectance and NDVI review using QGIS Python."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    Qgis, QgsApplication, QgsContrastEnhancement, QgsLayoutExporter, QgsLayoutItemLabel,
    QgsLayoutItemMap, QgsLayoutPoint, QgsLayoutSize, QgsPalettedRasterRenderer, QgsPrintLayout,
    QgsProject, QgsRasterLayer, QgsReferencedRectangle, QgsSingleBandGrayRenderer,
)

from build_spring_review_qgis import ndvi_renderer

YEAR = 2005
REVIEW_DAYS = ("2005-03-01", "2005-04-18", "2005-04-19", "2005-07-23")
REFLECTANCE = ("Landsat Red", "Landsat NIR", "MODIS Red", "MODIS NIR")
SOURCE_COLORS = (
    (1, "#427baa", "Observed Landsat"), (2, "#187963", "Two-pair fusion"),
    (3, "#f5b547", "Single-pair fusion"), (4, "#8e639c", "Landsat interpolation"),
)


def style(layer, name, band):
    provider = layer.dataProvider()
    if name in REFLECTANCE:
        high = 0.30 if name.endswith("Red") else 0.55
        stretch = QgsContrastEnhancement(provider.dataType(band))
        stretch.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
        stretch.setMinimumValue(0)
        stretch.setMaximumValue(high)
        renderer = QgsSingleBandGrayRenderer(provider, band)
        renderer.setContrastEnhancement(stretch)
    elif name == "NDVI source":
        classes = [QgsPalettedRasterRenderer.Class(value, QColor(color), label)
                   for value, color, label in SOURCE_COLORS]
        renderer = QgsPalettedRasterRenderer(provider, band, classes)
    else:
        renderer = ndvi_renderer(layer, (0.05, 0.90), band)
    layer.setRenderer(renderer)


def add_comparison(project, layers, extent):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("April and July comparison")
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(300, 150))

    def label(value, x, y, width, size=10, center=False):
        item = QgsLayoutItemLabel(layout)
        item.setText(value)
        item.setFont(QFont("Arial", size, QFont.Bold))
        item.setHAlign(Qt.AlignHCenter if center else Qt.AlignLeft)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(width, 8))

    label("Clayton Township  |  Landsat and MODIS, 2005", 7, 4, 285, 17)
    label("Shared display ranges  /  Red 0–0.30  /  NIR 0–0.55  /  NDVI 0.05–0.90", 7, 14, 285)
    for row, day in enumerate(("2005-04-18", "2005-07-23")):
        top = 25 + 57 * row
        label(day, 7, top, 285, 11)
        for index, name in enumerate((*REFLECTANCE, "Observed Landsat NDVI")):
            x = 7 + 58 * index
            label(name, x, top + 7, 54, center=True)
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(x + 6, top + 15))
            panel.attemptResize(QgsLayoutSize(42, 42))
            panel.setCrs(project.crs())
            panel.setLayers([layers[day, name]])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(extent)
            panel.setFrameEnabled(True)
    label("MODIS reflectance is resampled from the nominal 500 m product.", 7, 140, 285, 9)
    project.layoutManager().addLayout(layout)
    return layout


def build(root):
    project = QgsProject.instance()
    project.setTitle("Clayton Township | Historical inputs, 2005")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    reflectance = root / "reflectance" / str(YEAR)
    ndvi = root / "with_fallback" / str(YEAR)
    files = {name: reflectance / (f"{name.replace(' ', '_')}_{YEAR}.tif") for name in REFLECTANCE}
    files.update({
        "Observed Landsat NDVI": ndvi / f"Landsat_NDVI_{YEAR}.tif",
        "MODIS NDVI": ndvi / f"MODIS_NDVI_{YEAR}.tif",
        "Modeled NDVI": ndvi / f"NDVI_{YEAR}.tif",
        "NDVI source": ndvi / f"NDVI_source_{YEAR}.tif",
    })
    layers = {}
    selector = project.layerTreeRoot().addGroup("Select a date and one layer")
    for day in REVIEW_DAYS:
        band = (date.fromisoformat(day) - date(YEAR, 3, 1)).days + 1
        group = selector.addGroup(day)
        for name, path in files.items():
            layer = QgsRasterLayer(str(path), name)
            if not layer.isValid() or layer.bandCount() != 245 or day not in layer.bandName(band):
                raise ValueError(f"Missing date band {day} in {path}")
            style(layer, name, band)
            project.addMapLayer(layer, False)
            group.addLayer(layer).setExpanded(False)
            layers[day, name] = layer
        group.setIsMutuallyExclusive(True, 1)
        group.setExpanded(day == "2005-04-18")
    selector.setIsMutuallyExclusive(True, 1)
    base = layers["2005-04-18", "Landsat NIR"]
    extent = base.extent()
    project.setCrs(base.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, base.crs()))
    layout = add_comparison(project, layers, extent)
    target = root / "Clayton_2005_review.qgz"
    if not project.write(str(target)):
        raise RuntimeError(project.error())
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 160
    if QgsLayoutExporter(layout).exportToImage(str(root / "Clayton_2005_comparison.png"), settings) != QgsLayoutExporter.Success:
        raise RuntimeError("Could not render the April and July comparison")
    project.clear()
    if not project.read(str(target)) or len(project.mapLayers()) != len(layers) or not all(
        layer.isValid() for layer in project.mapLayers().values()
    ):
        raise RuntimeError(f"Could not reopen the comparison at {target}")
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
