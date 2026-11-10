#!/usr/bin/env python3
"""Use QGIS Python to build a shared-stretch project from a fusion pilot run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont
from qgis.core import (
    Qgis, QgsApplication, QgsLayoutItemLabel, QgsLayoutItemMap, QgsLayoutPoint,
    QgsLayoutSize, QgsPrintLayout, QgsProject, QgsRasterLayer, QgsReferencedRectangle,
)

from build_stair_qgis import fcc_renderer, renderer_matches


def build(directory):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    days = report["days"]
    truth_days = report["benchmark_dates"]
    project = QgsProject.instance()
    project.setTitle(f"Clayton fusion | {days[0]} to {days[-1]}")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    group = project.layerTreeRoot().addGroup("Select one layer to inspect")
    layers = {}
    base = None
    series = {"MODIS": "MODIS_aligned", "Fusion": "STAIR_fusion", "Landsat truth": "Landsat_truth"}
    for day in days:
        for label, stem in series.items():
            if label == "Landsat truth" and day not in truth_days:
                continue
            layer = QgsRasterLayer(str(directory / f"{stem}_{day}.tif"), f"{day} | {label}")
            if not layer.isValid() or layer.bandCount() != 3:
                raise ValueError(f"Invalid three-band layer: {layer.source()}")
            if base is None:
                base = layer
            if (layer.crs(), layer.extent(), layer.width(), layer.height()) != (
                base.crs(), base.extent(), base.width(), base.height()
            ):
                raise ValueError(f"Layer does not match the comparison grid: {layer.source()}")
            layer.setRenderer(fcc_renderer(layer))
            project.addMapLayer(layer, False)
            group.addLayer(layer).setExpanded(False)
            layers[day, label] = layer

    active = list(layers).index((truth_days[0], "Fusion"))
    group.setIsMutuallyExclusive(True, active)
    extent = base.extent()
    project.setCrs(base.crs())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, base.crs()))

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("FCC week comparison")
    layout.pageCollection().pages()[0].setPageSize(QgsLayoutSize(335, 165))

    def label(value, x, y, width, size=10, center=False):
        item = QgsLayoutItemLabel(layout)
        item.setText(value)
        item.setFont(QFont("Arial", size, QFont.Bold))
        if center:
            item.setHAlign(Qt.AlignHCenter)
        layout.addLayoutItem(item)
        item.attemptMove(QgsLayoutPoint(x, y))
        item.attemptResize(QgsLayoutSize(width, 8))

    label(project.title(), 6, 4, 320, size=17)
    label("Shared FCC  |  NIR 0.08–0.60  /  Red 0.005–0.165  /  Green 0.01–0.15", 6, 13, 320)
    for index, day in enumerate(days):
        x = 41 + 41 * index
        label(day, x, 22, 39, center=True)
        for row, series_label in enumerate(series):
            layer = layers.get((day, series_label))
            if layer is None:
                continue
            panel = QgsLayoutItemMap(layout)
            layout.addLayoutItem(panel)
            panel.attemptMove(QgsLayoutPoint(x, 30 + 41 * row))
            panel.attemptResize(QgsLayoutSize(39, 39))
            panel.setCrs(project.crs())
            panel.setLayers([layer])
            panel.setKeepLayerSet(True)
            panel.zoomToExtent(extent)
            panel.setFrameEnabled(True)
    for row, series_label in enumerate(series):
        label(series_label, 6, 45 + 41 * row, 35)
    label("Landsat in the third row was withheld from fusion and used for validation.", 6, 155, 320)
    project.layoutManager().addLayout(layout)
    target = directory / "fusion_week.qgz"
    if not project.write(str(target)):
        raise RuntimeError(f"Could not write {target}: {project.error()}")
    count = len(layers)
    project.clear()
    if not project.read(str(target)) or len(project.mapLayers()) != count or not all(
        layer.isValid() and renderer_matches(layer.renderer(), True) for layer in project.mapLayers().values()
    ):
        raise RuntimeError(f"Could not reopen all {count} FCC layers in {target}")
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    app = QgsApplication([], True)
    app.initQgis()
    build(args.directory.resolve())
