'use strict';
/**
 * build_all.js  —  生成 4 页可编辑 PPTX
 *
 * Slide 1: bijian    heatmap 比较 (GT / MeshMonk / ICP × 1-8 mm)
 * Slide 2: kedian    heatmap 比较
 * Slide 3: xiahedian heatmap 比较
 * Slide 4: 3D 脸模版页 (Template.obj + bijian/kedian/xiahedian × 1-8 mm，纯色)
 *
 * 运行: node build_all.js
 */

const PptxGenJS  = require('pptxgenjs');
const path       = require('path');
const fs         = require('fs');
const { execSync } = require('child_process');

// ── 路径配置 ────────────────────────────────────────────────────────────────
const PPTX_DIR   = __dirname;
const OUTPUT     = path.resolve(PPTX_DIR, 'all_comparison.pptx');

// ── 数据配置 ────────────────────────────────────────────────────────────────
const DISPLACEMENTS = [1, 2, 3, 4, 5, 6, 7, 8];

const HEATMAP_ROWS = [
    { name: 'gt',  label: 'Ground Truth' },
    { name: 'mm',  label: 'MeshMonk'     },
    { name: 'icp', label: 'ICP'          },
];

const DATASETS = [
    { name: 'bijian',    label: 'Bijian'    },
    { name: 'kedian',    label: 'Kedian'    },
    { name: 'xiahedian', label: 'Xiahedian' },
];

// ── 布局常量（LAYOUT_16x9: 10 × 5.625 英寸）────────────────────────────────
const SW     = 10.0,  SH     = 5.625;
const MARGIN = 0.08,  LBLW   = 0.55;
const HDRY   = 0.28,  GAP_C  = 0.02,  GAP_R = 0.04;
const nC     = DISPLACEMENTS.length;  // 8

// ── 工具函数 ────────────────────────────────────────────────────────────────
function getImgDims(imgPath) {
    const out = execSync(`sips -g pixelWidth -g pixelHeight "${imgPath}"`).toString();
    const w   = parseInt(out.match(/pixelWidth:\s*(\d+)/)[1]);
    const h   = parseInt(out.match(/pixelHeight:\s*(\d+)/)[1]);
    return [w, h];
}

function firstExisting(paths) {
    return paths.find(p => fs.existsSync(p)) || null;
}

// ── Slide 1-3: heatmap 比较页 ────────────────────────────────────────────────
function addHeatmapSlide(pptx, datasetName, datasetLabel) {
    const cellsDir = path.join(PPTX_DIR, datasetName);

    // 找参考图获取宽高比
    const refImg = firstExisting(
        DISPLACEMENTS.flatMap(d => HEATMAP_ROWS.map(r =>
            path.join(cellsDir, `${d}_${r.name}.png`)))
    );
    if (!refImg) {
        console.warn(`  跳过 ${datasetLabel}: 未找到 PNG cells`);
        return;
    }

    const [pxW, pxH] = getImgDims(refImg);
    const aspect     = pxW / pxH;
    const nR         = HEATMAP_ROWS.length;
    const cellW      = (SW - 2*MARGIN - LBLW - GAP_C*(nC-1)) / nC;
    const cellH      = (SH - 2*MARGIN - HDRY - GAP_R*(nR-1)) / nR;
    const imgW       = cellW;
    const imgH       = imgW / aspect;

    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };

    // 列标题
    DISPLACEMENTS.forEach((d, ci) => {
        slide.addText(`${d} mm`, {
            x: MARGIN + LBLW + ci*(cellW + GAP_C),
            y: MARGIN, w: cellW, h: HDRY,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 11, bold: true, color: '000000',
        });
    });

    // 行标签 + 图片
    HEATMAP_ROWS.forEach(({ name, label }, ri) => {
        const rowY = MARGIN + HDRY + ri*(cellH + GAP_R);

        slide.addText(label, {
            x: MARGIN, y: rowY + (cellH - 0.28)/2, w: LBLW, h: 0.28,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 8, bold: true, color: '000000',
        });

        DISPLACEMENTS.forEach((d, ci) => {
            const imgPath = path.join(cellsDir, `${d}_${name}.png`);
            if (!fs.existsSync(imgPath)) { console.warn(`  跳过: ${imgPath}`); return; }
            slide.addImage({
                path: imgPath,
                x: MARGIN + LBLW + ci*(cellW + GAP_C),
                y: rowY + (cellH - imgH)/2,
                w: imgW, h: imgH,
            });
        });
    });

    console.log(`✓ Slide: ${datasetLabel} heatmap`);
}

// ── Slide 4: 3D 脸模版页 ────────────────────────────────────────────────────
function addTemplatePage(pptx) {
    const tplDir      = path.join(PPTX_DIR, 'template_page');
    const templatePng = path.join(tplDir, 'template.png');

    if (!fs.existsSync(templatePng)) {
        console.warn('  template.png 不存在，跳过 template page');
        return;
    }

    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };

    // ── 左侧：Template.obj ───────────────────────────────────────────────────
    const TPL_COL_W = 1.4;   // 左侧列宽（英寸）
    const GRID_GAP  = 0.10;  // 左侧与右侧网格的间距

    const [tpxW, tpxH] = getImgDims(templatePng);
    const tAspect       = tpxW / tpxH;
    // 左侧图片：宽度自适应 TPL_COL_W，高度受 slide 高度约束
    const tImgH = Math.min(SH - 2*MARGIN - 0.25, TPL_COL_W / tAspect);
    const tImgW = tImgH * tAspect;
    const tImgX = MARGIN + (TPL_COL_W - tImgW) / 2;
    const tImgY = MARGIN + 0.25 + (SH - 2*MARGIN - 0.25 - tImgH) / 2;

    slide.addImage({ path: templatePng, x: tImgX, y: tImgY, w: tImgW, h: tImgH });

    // "Template" 标签（左列顶部）
    slide.addText('Template', {
        x: MARGIN, y: MARGIN, w: TPL_COL_W, h: 0.22,
        align: 'center', valign: 'middle',
        fontFace: 'Times New Roman', fontSize: 9, bold: true, color: '000000',
    });

    // ── 右侧：3 行 × 8 列纯色网格 ────────────────────────────────────────────
    const gridX0  = MARGIN + TPL_COL_W + GRID_GAP;
    const gridW   = SW - gridX0 - MARGIN;
    const pCellW  = (gridW - LBLW - GAP_C*(nC-1)) / nC;
    const nR      = DATASETS.length;
    const pCellH  = (SH - 2*MARGIN - HDRY - GAP_R*(nR-1)) / nR;

    // 获取纯色网格图的宽高比（取第一张存在的图）
    const refPlain = firstExisting(
        DATASETS.flatMap(ds => DISPLACEMENTS.map(d =>
            path.join(tplDir, `${ds.name}_${d}.png`)))
    );
    let pImgW = pCellW, pImgH = pCellH;  // fallback
    if (refPlain) {
        const [ppxW, ppxH] = getImgDims(refPlain);
        pImgW = pCellW;
        pImgH = pImgW / (ppxW / ppxH);
    }

    // 列标题
    DISPLACEMENTS.forEach((d, ci) => {
        slide.addText(`${d} mm`, {
            x: gridX0 + LBLW + ci*(pCellW + GAP_C),
            y: MARGIN, w: pCellW, h: HDRY,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 10, bold: true, color: '000000',
        });
    });

    // 行标签 + 图片
    DATASETS.forEach(({ name, label }, ri) => {
        const rowY = MARGIN + HDRY + ri*(pCellH + GAP_R);

        slide.addText(label, {
            x: gridX0, y: rowY + (pCellH - 0.26)/2, w: LBLW, h: 0.26,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 8, bold: true, color: '000000',
        });

        DISPLACEMENTS.forEach((d, ci) => {
            const imgPath = path.join(tplDir, `${name}_${d}.png`);
            if (!fs.existsSync(imgPath)) { console.warn(`  跳过: ${imgPath}`); return; }
            slide.addImage({
                path: imgPath,
                x: gridX0 + LBLW + ci*(pCellW + GAP_C),
                y: rowY + (pCellH - pImgH)/2,
                w: pImgW, h: pImgH,
            });
        });
    });

    console.log(`✓ Slide: template page`);
}

// ── 主函数 ──────────────────────────────────────────────────────────────────
async function build() {
    const pptx  = new PptxGenJS();
    pptx.layout = 'LAYOUT_16x9';

    DATASETS.forEach(({ name, label }) => addHeatmapSlide(pptx, name, label));
    addTemplatePage(pptx);

    await pptx.writeFile({ fileName: OUTPUT });
    console.log(`\n✓ PPTX 已保存: ${OUTPUT}`);
}

build().catch(err => { console.error(err); process.exit(1); });
