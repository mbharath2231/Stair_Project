import re

# ==============================================================================
# 1. CLEAN STAIR 2x2 FRAMEWORK (outputs/stair_drawio_diagram.drawio)
# ==============================================================================
drawio_2x2 = """<mxfile host="app.diagrams.net" modified="2026-11-03T14:00:00.000Z" agent="Mozilla/5.0" version="24.0.0">
  <diagram id="stair_diagram" name="STAIR_2x2_Framework">
    <mxGraphModel dx="1400" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1250" pageHeight="1020" background="#0F172A" math="0" shadow="1">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- TITLE BAR -->
        <mxCell id="title" value="&lt;b&gt;&lt;font style=&quot;font-size: 22px;&quot; color=&quot;#FFFFFF&quot;&gt;STAIR Spatiotemporal Fusion Framework (Luo et al., 2018)&lt;/font&gt;&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size: 13px;&quot; color=&quot;#94A3B8&quot;&gt;Balanced 2×2 Methodological Architecture for Cloud Imputation and Daily High-Resolution Agricultural Synthesis&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="60" y="25" width="1120" height="55" as="geometry" />
        </mxCell>

        <!-- QUADRANT 1: TOP-LEFT (DATA ACQUISITION) -->
        <mxCell id="q1" value="&lt;b&gt;1. DATA ACQUISITION &amp;amp; QUALITY SCREENING&lt;/b&gt;&lt;br&gt;&lt;br&gt;
&lt;b&gt;Primary Satellite Observations:&lt;/b&gt;&lt;br&gt;
• Landsat 8 &amp;amp; 9: 30 m · 16-Day Revisit (NIR, Red, Green)&lt;br&gt;
• MODIS MCD43A4: 500 m · Daily Coverage (Nadir BRDF)&lt;br&gt;
• Clear Reference Scene (t₁): Cloud-free structural baseline&lt;br&gt;
• Full-Season Archive: Multi-temporal training catalog&lt;br&gt;
• Study Area AOI: 5 km × 5 km Extent (Clayton Township, MI)&lt;br&gt;&lt;br&gt;
&lt;b&gt;Quality Screening &amp;amp; Water Guard:&lt;/b&gt;&lt;br&gt;
• QA_PIXEL Bitmasks: Cloud, Shadow, Dilated &amp;amp; Cirrus flags&lt;br&gt;
• 6-Pixel Buffer Dilation to catch diffuse haze &amp;amp; shadow edges&lt;br&gt;
• JRC Global Surface Water + NDWI Guard: Protects lakes&lt;br&gt;&lt;br&gt;
&lt;b&gt;Quality Partitioned Layers:&lt;/b&gt;&lt;br&gt;
• Clear Surface Mask (C_f): Valid calibration pixels&lt;br&gt;
• Cloud Gap Mask (C_g): Missing target pixels to restore" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFBEB;strokeColor=#D97706;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=16;spacingRight=16;spacingBottom=16;fontSize=11.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;arcSize=4;" vertex="1" parent="1">
          <mxGeometry x="60" y="100" width="480" height="370" as="geometry" />
        </mxCell>

        <!-- QUADRANT 2: TOP-RIGHT (CLOUD IMPUTATION) -->
        <mxCell id="q2" value="&lt;b&gt;2. STAIR CLOUD IMPUTATION PIPELINE (SEC. 2.2)&lt;/b&gt;&lt;br&gt;&lt;br&gt;
&lt;b&gt;Stage 2.2.1: Temporal Linear Interpolation:&lt;/b&gt;&lt;br&gt;
• Nearest clear observations before (t_bef) and after (t_aft)&lt;br&gt;
• Proportional weight: w = (t₀ − t_bef) / (t_aft − t_bef)&lt;br&gt;
• Formula: L_linear = R(t_bef) + w × [ R(t_aft) − R(t_bef) ]&lt;br&gt;
• Forward/backward fill fallback for season overpass boundaries&lt;br&gt;&lt;br&gt;
&lt;b&gt;Stage 2.2.2a: Global Spatial Drift Correction:&lt;/b&gt;&lt;br&gt;
• Temporal drift: Δ_temp = mean(L, C_f, t₀) − mean(L_linear, C_f, t₀)&lt;br&gt;
• Spatial gap offset: Δ_spatial = mean(L, C_g, t₁) − mean(L, C_f, t₁)&lt;br&gt;
• Formula: L_global = L_linear + (Δ_temp + Δ_spatial)&lt;br&gt;&lt;br&gt;
&lt;b&gt;Stage 2.2.2b: Land-Cover Adaptive K-Means Correction:&lt;/b&gt;&lt;br&gt;
• Unsupervised Weka K-Means (k = 4) clustering on clear day t₁&lt;br&gt;
• Class-specific offsets: Δ_temp(c) &amp;amp; Δ_spatial(c) per crop type&lt;br&gt;
• 1.5 px focal mean boundary feathering eliminates seamlines&lt;br&gt;
• Formula: L_adaptive = L_linear + δ(c)&lt;br&gt;&lt;br&gt;
&lt;b&gt;Output:&lt;/b&gt; ✓ 100% Cloud-Free Landsat Image (30 m Resolution)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ECFEFF;strokeColor=#0891B2;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=16;spacingRight=16;spacingBottom=16;fontSize=11.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;arcSize=4;" vertex="1" parent="1">
          <mxGeometry x="700" y="100" width="480" height="370" as="geometry" />
        </mxCell>

        <!-- QUADRANT 3: BOTTOM-LEFT (SPATIOTEMPORAL FUSION) -->
        <mxCell id="q3" value="&lt;b&gt;3. SPATIOTEMPORAL DUAL FUSION ENGINE (SEC. 2.3)&lt;/b&gt;&lt;br&gt;&lt;br&gt;
&lt;b&gt;Dual Reference Pair Selection:&lt;/b&gt;&lt;br&gt;
• Golden Day (T₁): Concurrent clear Landsat (30 m) &amp;amp; MODIS (500 m)&lt;br&gt;
• Post-Reference (T₂): Imputed Landsat (30 m) &amp;amp; MODIS (500 m)&lt;br&gt;&lt;br&gt;
&lt;b&gt;Spatial Scale &amp;amp; Grid Alignment:&lt;/b&gt;&lt;br&gt;
• Bilinear interpolation resamples 500 m MODIS to 30 m grid&lt;br&gt;
• Aligns coordinates, projections, and pixel bounds to Landsat grid&lt;br&gt;&lt;br&gt;
&lt;b&gt;Spatial Difference Residual Isolation:&lt;/b&gt;&lt;br&gt;
• ε(T₁) = L(T₁) − M(T₁) &amp;amp; ε(T₂) = L̂(T₂) − M(T₂)&lt;br&gt;
• Extracts 30 m farm field boundaries &amp;amp; crop rows blurred at 500 m&lt;br&gt;&lt;br&gt;
&lt;b&gt;Proximity-Weighted Daily Synthesis:&lt;/b&gt;&lt;br&gt;
• Time weights: w₁ = (T₂ − t_p) / (T₂ − T₁) ; w₂ = (t_p − T₁) / (T₂ − T₁)&lt;br&gt;
• Formula: I(t_p) = M(t_p) + w₁·ε(T₁) + w₂·ε(T₂)&lt;br&gt;
• Clamped to valid physical reflectance range [0.0, 1.0]&lt;br&gt;&lt;br&gt;
&lt;b&gt;Output:&lt;/b&gt; ✓ Fused 30 m Daily Synthetic Product (Daily Phenology)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#7C3AED;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=16;spacingRight=16;spacingBottom=16;fontSize=11.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;arcSize=4;" vertex="1" parent="1">
          <mxGeometry x="60" y="580" width="480" height="370" as="geometry" />
        </mxCell>

        <!-- QUADRANT 4: BOTTOM-RIGHT (VERIFICATION & DELIVERABLES) -->
        <mxCell id="q4" value="&lt;b&gt;4. QUANTITATIVE VERIFICATION &amp;amp; DELIVERABLES&lt;/b&gt;&lt;br&gt;&lt;br&gt;
&lt;b&gt;Ground-Truth Benchmark Proof:&lt;/b&gt;&lt;br&gt;
• 1.5 km simulated cloud patch over 0% clear benchmark&lt;br&gt;
• Multi-season validation across 6 phenological phases (2022)&lt;br&gt;&lt;br&gt;
&lt;b&gt;Quantitative Accuracy Assessment:&lt;/b&gt;&lt;br&gt;
• Evaluated inside cloud gap: MAE, RMSE, Pearson Correlation (R), R²&lt;br&gt;
• Documented Gain: 20% to 40% error reduction over baseline&lt;br&gt;&lt;br&gt;
&lt;b&gt;Geospatial Deliverables &amp;amp; Research Applications:&lt;/b&gt;&lt;br&gt;
• Multi-band calibrated GeoTIFF stacks (EPSG:32616)&lt;br&gt;
• Continuous daily NDVI agricultural time-series&lt;br&gt;
• Ready for GIS ingestion in QGIS / ArcGIS&lt;br&gt;&lt;br&gt;
&lt;b&gt;Flow Legend:&lt;/b&gt;&lt;br&gt;
[Amber: Inputs] · [Cyan: Impute] · [Purple: Fusion] · [Green: Deliverables]" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#16A34A;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=16;spacingRight=16;spacingBottom=16;fontSize=11.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;arcSize=4;" vertex="1" parent="1">
          <mxGeometry x="700" y="580" width="480" height="370" as="geometry" />
        </mxCell>

        <!-- ARROW 1: Q1 -> Q2 (Top Horizontal Channel) -->
        <mxCell id="edge1" value="Feeds Masked Target &amp;amp; Quality Masks" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;strokeColor=#0891B2;strokeWidth=2.5;fontSize=11;fontColor=#0E7490;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="q1" target="q2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ARROW 2: Q1 -> Q3 (Left Vertical Channel) -->
        <mxCell id="edge2" value="Supplies Daily MODIS M(t_p)" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.25;entryY=0;entryDx=0;entryDy=0;exitX=0.25;exitY=1;exitDx=0;exitDy=0;dashed=1;strokeColor=#7C3AED;strokeWidth=2;fontSize=11;fontColor=#6D28D9;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="q1" target="q3">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ARROW 3: Q2 -> Q3 (Routed via Central Runway, No Overlapping Diagonal) -->
        <mxCell id="edge3" value="Supplies Imputed Landsat T₂ Reference" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.75;entryY=0;entryDx=0;entryDy=0;exitX=0.25;exitY=1;exitDx=0;exitDy=0;strokeColor=#7C3AED;strokeWidth=2.5;fontSize=11;fontColor=#6D28D9;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="q2" target="q3">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="820" y="525" />
              <mxPoint x="420" y="525" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ARROW 4: Q3 -> Q4 (Bottom Horizontal Channel) -->
        <mxCell id="edge4" value="Synthesized 30m Product for Verification" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;strokeColor=#16A34A;strokeWidth=2.5;fontSize=11;fontColor=#15803D;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="q3" target="q4">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

with open('outputs/stair_drawio_diagram.drawio', 'w') as f:
    f.write(drawio_2x2)

print("Updated outputs/stair_drawio_diagram.drawio successfully!")

# ==============================================================================
# 2. CLEAN STAIR DETAILED FLOWCHART (outputs/stair_detailed_flowchart.drawio)
# ==============================================================================
drawio_detailed = """<mxfile host="app.diagrams.net" modified="2026-11-03T14:00:00.000Z" agent="Mozilla/5.0" version="24.0.0">
  <diagram id="stair_detailed" name="STAIR_Detailed_Flowchart">
    <mxGraphModel dx="1600" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1540" pageHeight="1280" background="#0F172A" math="0" shadow="1">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- MASTER TITLE -->
        <mxCell id="title" value="&lt;b&gt;&lt;font style=&quot;font-size: 22px;&quot; color=&quot;#FFFFFF&quot;&gt;STAIR Spatiotemporal Fusion Framework (Luo et al., 2018)&lt;/font&gt;&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size: 13px;&quot; color=&quot;#94A3B8&quot;&gt;Detailed End-to-End Methodological Architecture: Satellite Data Intake → Multi-Scale Imputation → Spatiotemporal Fusion → Validation&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="60" y="25" width="1420" height="60" as="geometry" />
        </mxCell>

        <!-- ========================================== -->
        <!-- FRAME 1: DATA ACQUISITION & QUALITY SCREENING -->
        <!-- ========================================== -->
        <mxCell id="f1" value="1. DATA ACQUISITION &amp;amp; QUALITY SCREENING" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#FFFBEB;strokeColor=#D97706;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;fontFamily=Helvetica;fontSize=13;fontColor=#92400E;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="60" y="110" width="590" height="490" as="geometry" />
        </mxCell>

        <mxCell id="101" value="Landsat 8 &amp;amp; 9&lt;br&gt;30m · 16-Day Revisit" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="160" width="165" height="44" as="geometry" />
        </mxCell>

        <mxCell id="102" value="MODIS MCD43A4&lt;br&gt;500m · Daily Nadir BRDF" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="270" y="160" width="170" height="44" as="geometry" />
        </mxCell>

        <mxCell id="103" value="Reference Scene (t₁)&lt;br&gt;Clear Structural Baseline" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="460" y="160" width="165" height="44" as="geometry" />
        </mxCell>

        <mxCell id="104" value="Pixel Quality Screening&lt;br&gt;QA_PIXEL bitmasks: cloud, shadow, cirrus &amp;amp; dilated flags" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E0E7FF;strokeColor=#4F46E5;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="235" width="540" height="42" as="geometry" />
        </mxCell>

        <mxCell id="105" value="6-Pixel Buffer Dilation&lt;br&gt;Captures diffuse cloud &amp;amp; shadow edges" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E0E7FF;strokeColor=#4F46E5;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="305" width="260" height="44" as="geometry" />
        </mxCell>

        <mxCell id="106" value="Permanent Water Guard&lt;br&gt;JRC Global Water + NDWI protects lakes" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E0E7FF;strokeColor=#4F46E5;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="365" y="305" width="260" height="44" as="geometry" />
        </mxCell>

        <mxCell id="107" value="Clear Surface Mask (C_f)&lt;br&gt;Valid unclouded calibration pixels" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#065F46;fontFamily=Helvetica;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="380" width="260" height="48" as="geometry" />
        </mxCell>

        <mxCell id="108" value="Cloud Gap Mask (C_g)&lt;br&gt;Missing cloudy pixels to restore" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#DC2626;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#991B1B;fontFamily=Helvetica;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="365" y="380" width="260" height="48" as="geometry" />
        </mxCell>

        <mxCell id="109" value="Domain: Clayton Township, MI (5 km × 5 km Agricultural Chip, EPSG:32616)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#B45309;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="460" width="540" height="34" as="geometry" />
        </mxCell>

        <mxCell id="110" value="Partitions scene into clear calibration pixels (C_f) and cloudy restoration gap (C_g)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#92400E;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="510" width="540" height="32" as="geometry" />
        </mxCell>

        <!-- Edges inside Frame 1 -->
        <mxCell id="111" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#D97706;strokeWidth=1.5;" edge="1" parent="1" source="101" target="104"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="112" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#D97706;strokeWidth=1.5;" edge="1" parent="1" source="102" target="104"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="113" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#D97706;strokeWidth=1.5;" edge="1" parent="1" source="103" target="104"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="114" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#4F46E5;strokeWidth=1.5;" edge="1" parent="1" source="104" target="105"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="115" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#4F46E5;strokeWidth=1.5;" edge="1" parent="1" source="105" target="106"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="116" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#059669;strokeWidth=1.5;" edge="1" parent="1" source="106" target="107"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="117" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#DC2626;strokeWidth=1.5;" edge="1" parent="1" source="106" target="108"><mxGeometry relative="1" as="geometry" /></mxCell>

        <!-- ========================================== -->
        <!-- FRAME 2: MULTI-SCALE CLOUD IMPUTATION PIPELINE -->
        <!-- ========================================== -->
        <mxCell id="f2" value="2. MULTI-SCALE CLOUD IMPUTATION PIPELINE (SEC. 2.2)" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#ECFEFF;strokeColor=#0891B2;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;fontFamily=Helvetica;fontSize=13;fontColor=#0E7490;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="810" y="110" width="590" height="490" as="geometry" />
        </mxCell>

        <mxCell id="118" value="2.1 Nearest Clear Search&lt;br&gt;Closest scenes: t_bef &amp;amp; t_aft" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="160" width="250" height="44" as="geometry" />
        </mxCell>

        <mxCell id="119" value="Temporal Linear Interpolation&lt;br&gt;L_linear = R(t_bef) + w·[R(t_aft) − R(t_bef)]" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1125" y="160" width="250" height="44" as="geometry" />
        </mxCell>

        <mxCell id="120" value="2.2 Temporal Drift Measurement&lt;br&gt;Δ_temp = mean(C_f, t₀) − mean(L_linear, C_f)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="235" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="121" value="Global Spatial Drift Offset&lt;br&gt;L_global = L_linear + (Δ_temp + Δ_spatial)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1125" y="235" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="122" value="2.3 Weka K-Means Clustering&lt;br&gt;(k = 4) groups crop types on clear day t₁" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="315" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="123" value="Land-Cover Adaptive Calibration&lt;br&gt;Class-specific shift δ(c) + 1.5 px feathering" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1125" y="315" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="124" value="✓ 100% Cloud-Free Landsat Image (30 m Resolution)&lt;br&gt;&lt;font style=&quot;font-size: 9.5px; font-weight: normal;&quot;&gt;Zero Seamlines · All farm parcel boundaries preserved&lt;/font&gt;" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#06B6D4;strokeColor=#0891B2;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;fontColor=#FFFFFF;fontFamily=Helvetica;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="395" width="540" height="50" as="geometry" />
        </mxCell>

        <mxCell id="125" value="Parallel MODIS Imputation: Daily 500m MODIS also made 100% cloud-free" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ECFEFF;strokeColor=#06B6D4;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#0E7490;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="465" width="540" height="34" as="geometry" />
        </mxCell>

        <mxCell id="126" value="Resolves atmospheric drift and intra-class phenological differences seamlessly" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#CFFAFE;strokeColor=#0891B2;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#0E7490;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="515" width="540" height="32" as="geometry" />
        </mxCell>

        <!-- Edges inside Frame 2 -->
        <mxCell id="127" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=1.5;" edge="1" parent="1" source="118" target="119"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="128" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=1.5;" edge="1" parent="1" source="119" target="120"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="129" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=1.5;" edge="1" parent="1" source="120" target="121"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="130" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=1.5;" edge="1" parent="1" source="121" target="122"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="131" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=1.5;" edge="1" parent="1" source="122" target="123"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="132" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0891B2;strokeWidth=2;" edge="1" parent="1" source="123" target="124"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="133" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#06B6D4;strokeWidth=1.5;" edge="1" parent="1" source="124" target="125"><mxGeometry relative="1" as="geometry" /></mxCell>

        <!-- ========================================== -->
        <!-- FRAME 3: SPATIOTEMPORAL DUAL FUSION ENGINE -->
        <!-- ========================================== -->
        <mxCell id="f3" value="3. SPATIOTEMPORAL DUAL FUSION ENGINE (SEC. 2.3)" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#7C3AED;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;fontFamily=Helvetica;fontSize=13;fontColor=#6D28D9;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="60" y="700" width="590" height="490" as="geometry" />
        </mxCell>

        <mxCell id="134" value="Dual Reference Pairing&lt;br&gt;Golden Day (T₁) &amp;amp; Post-Ref (T₂ Imputed)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="750" width="260" height="46" as="geometry" />
        </mxCell>

        <mxCell id="135" value="Scale &amp;amp; Grid Alignment&lt;br&gt;Resamples 500m MODIS to 30m Landsat grid" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="365" y="750" width="260" height="46" as="geometry" />
        </mxCell>

        <mxCell id="136" value="Spatial Residual Isolation&lt;br&gt;ε(T) = Landsat(30m) − MODIS(30m)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="830" width="260" height="46" as="geometry" />
        </mxCell>

        <mxCell id="137" value="Intra-Parcel Structural Extraction&lt;br&gt;Captures 30m fence lines, roads &amp;amp; ditches" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="365" y="830" width="260" height="46" as="geometry" />
        </mxCell>

        <mxCell id="138" value="Proximity-Weighted Daily Synthesis&lt;br&gt;I_hat(t_p) = MODIS(t_p) + w₁·ε(T₁) + w₂·ε(T₂)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="910" width="540" height="46" as="geometry" />
        </mxCell>

        <mxCell id="139" value="✓ Fused 30 m Daily Synthetic Product&lt;br&gt;&lt;font style=&quot;font-size: 9.5px; font-weight: normal;&quot;&gt;Combines Daily Phenological Radiometry + Crisp 30m Field Geometry&lt;/font&gt;" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#8B5CF6;strokeColor=#7C3AED;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;fontColor=#FFFFFF;fontFamily=Helvetica;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="990" width="540" height="50" as="geometry" />
        </mxCell>

        <mxCell id="140" value="Physical reflectance clamped to [0.0, 1.0] · Cloud-free on any calendar day" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#8B5CF6;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#6D28D9;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="1065" width="540" height="34" as="geometry" />
        </mxCell>

        <mxCell id="141" value="Unmixes 500m pixels using high-frequency spatial blueprints from reference pairs" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#7C3AED;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#6D28D9;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="85" y="1115" width="540" height="32" as="geometry" />
        </mxCell>

        <!-- Edges inside Frame 3 -->
        <mxCell id="142" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=1.5;" edge="1" parent="1" source="134" target="135"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="143" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=1.5;" edge="1" parent="1" source="135" target="136"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="144" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=1.5;" edge="1" parent="1" source="136" target="137"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="145" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=1.5;" edge="1" parent="1" source="137" target="138"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="146" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=2;" edge="1" parent="1" source="138" target="139"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="147" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#8B5CF6;strokeWidth=1.5;" edge="1" parent="1" source="139" target="140"><mxGeometry relative="1" as="geometry" /></mxCell>

        <!-- ========================================== -->
        <!-- FRAME 4: QUANTITATIVE VERIFICATION & GIS DELIVERABLES -->
        <!-- ========================================== -->
        <mxCell id="f4" value="4. QUANTITATIVE VERIFICATION &amp;amp; GIS DELIVERABLES" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#16A34A;strokeWidth=2.5;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;fontFamily=Helvetica;fontSize=13;fontColor=#166534;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="810" y="700" width="590" height="490" as="geometry" />
        </mxCell>

        <mxCell id="148" value="Ground-Truth Benchmark Proof&lt;br&gt;1.5 km simulated cloud patch over clear day" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="750" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="149" value="Multi-Season Phenology Evaluation&lt;br&gt;Tested across 6 crop growth stages (2022)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1125" y="750" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="150" value="Statistical Verification&lt;br&gt;MAE, RMSE, Pearson (R), R² (20-40% Gain)" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="830" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="151" value="Absolute Error Heatmap (NIR)&lt;br&gt;Near-zero error (&amp;lt; 0.02) over farm parcels" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1125" y="830" width="250" height="46" as="geometry" />
        </mxCell>

        <mxCell id="152" value="GIS Deliverables: Calibrated Multi-Band GeoTIFFs&lt;br&gt;Native UTM Zone 16N / EPSG:32616 projection" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#111827;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="910" width="540" height="44" as="geometry" />
        </mxCell>

        <mxCell id="153" value="✓ Continuous Daily NDVI Time-Series for QGIS / ArcGIS&lt;br&gt;&lt;font style=&quot;font-size: 9.5px; font-weight: normal;&quot;&gt;Agricultural Monitoring Across Entire Growing Season&lt;/font&gt;" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#22C55E;strokeColor=#16A34A;strokeWidth=2;align=center;verticalAlign=middle;fontSize=11;fontColor=#FFFFFF;fontFamily=Helvetica;fontStyle=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="990" width="540" height="50" as="geometry" />
        </mxCell>

        <mxCell id="154" value="Status: Operational · Tested on 5km x 5km Contiguous Landscape &amp;amp; Farm Zooms" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#22C55E;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#166534;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="1065" width="540" height="34" as="geometry" />
        </mxCell>

        <mxCell id="155" value="Ready for direct drag-and-drop into QGIS with pre-configured False Color Composites" style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#16A34A;strokeWidth=1.5;align=center;verticalAlign=middle;fontSize=10.5;fontColor=#166534;fontFamily=Helvetica;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="835" y="1115" width="540" height="32" as="geometry" />
        </mxCell>

        <!-- Edges inside Frame 4 -->
        <mxCell id="156" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=1.5;" edge="1" parent="1" source="148" target="149"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="157" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=1.5;" edge="1" parent="1" source="149" target="150"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="158" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=1.5;" edge="1" parent="1" source="150" target="151"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="159" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=1.5;" edge="1" parent="1" source="151" target="152"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="160" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;" edge="1" parent="1" source="152" target="153"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="161" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#22C55E;strokeWidth=1.5;" edge="1" parent="1" source="153" target="154"><mxGeometry relative="1" as="geometry" /></mxCell>

        <!-- ============================================================= -->
        <!-- INTER-FRAME CLEAN CONNECTOR ARROWS (ROUTED THROUGH RUNWAYS)   -->
        <!-- ============================================================= -->

        <!-- ARROW 1: Box 1 (C_g Mask) -> Box 2 (Imputation Stage 2.1) -->
        <!-- Exits 108 right at (625, 404), enters central runway at x=730, goes up to y=182, enters 118 left -->
        <mxCell id="162" value="Feeds Cloud Gap C_g&lt;br&gt;into Imputation Engine" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0891B2;strokeWidth=2.5;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;fontSize=10;fontColor=#0E7490;fontFamily=Helvetica;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="108" target="118">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="730" y="404" />
              <mxPoint x="730" y="182" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ARROW 2: Box 1 (MODIS) -> Box 3 (Fusion Scale Alignment) -->
        <!-- Drops vertically through the 100 px horizontal runway between Tier 1 & 2 -->
        <mxCell id="163" value="Supplies Daily MODIS M(t_p)" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7C3AED;strokeWidth=2;dashed=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;fontSize=10;fontColor=#6D28D9;fontFamily=Helvetica;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="110" target="135">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="355" y="650" />
              <mxPoint x="495" y="650" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ARROW 3: Box 2 (Imputed Landsat 124) -> Box 3 (Fusion Dual Reference 134) -->
        <!-- Exits 124 left into central runway, drops to central horizontal runway y=650, travels left to x=215, enters 134 top -->
        <mxCell id="164" value="Supplies Imputed 100% Cloud-Free Landsat (30 m) as Fusion Reference Pair T₂" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7C3AED;strokeWidth=2.5;exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;fontSize=10;fontColor=#6D28D9;fontFamily=Helvetica;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="124" target="134">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="750" y="420" />
              <mxPoint x="750" y="650" />
              <mxPoint x="215" y="650" />
            </Array>
          </mxGeometry>
        </mxCell>

        <!-- ARROW 4: Box 3 (Fused 30m Product 139) -> Box 4 (Continuous Daily NDVI 153) -->
        <!-- Perfectly straight horizontal arrow from (625, 1015) to (835, 1015) across the 160 px runway -->
        <mxCell id="165" value="Synthesized 30m Product&lt;br&gt;for Continuous Daily NDVI" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#16A34A;strokeWidth=2.5;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;fontSize=10;fontColor=#15803D;fontFamily=Helvetica;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="139" target="153">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- ARROW 5: Box 3 (Fused 30m Product 139) -> Box 4 (Ground Truth Verification 148) -->
        <mxCell id="166" value="Supplies Product for Blind Validation" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#16A34A;strokeWidth=2;dashed=1;exitX=1;exitY=0.25;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;fontSize=10;fontColor=#15803D;fontFamily=Helvetica;fontStyle=1;labelBackgroundColor=#FFFFFF;" edge="1" parent="1" source="139" target="148">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="710" y="1002" />
              <mxPoint x="710" y="773" />
            </Array>
          </mxGeometry>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

with open('outputs/stair_detailed_flowchart.drawio', 'w') as f:
    f.write(drawio_detailed)

print("Updated outputs/stair_detailed_flowchart.drawio successfully!")
