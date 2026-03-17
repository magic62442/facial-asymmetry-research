'use strict';
/**
 * build_kedian.js
 * 用 PptxGenJS 将 kedian/ 目录下的 PNG cells 组合成可编辑的 PPTX。
 * 每张图（heatmap+colorbar）、每段文字都是独立元素，可在 PowerPoint 中自由移动/调整。
 *
 * 运行: node build_kedian.js
 */

const PptxGenJS = require('pptxgenjs');
const path      = require('path');
const fs        = require('fs');
const { execSync } = require('child_process');

// ── 配置 ────────────────────────────────────────────────────────────────────
const CELLS_DIR   = path.resolve(__dirname, 'kedian');
const OUTPUT_PATH = path.resolve(__dirname, 'kedian_comparison.pptx');

const DISPLACEMENTS = [1, 2, 3, 4, 5, 6, 7, 8];
const ROWS = [
    { name: 'gt',  label: 'Ground Truth' },
    { name: 'mm',  label: 'MeshMonk'     },
    { name: 'icp', label: 'ICP'          },
];

// ── 获取图片像素尺寸（macOS sips） ─────────────────────────────────────────
function getImgDims(imgPath) {
    const out = execSync(`sips -g pixelWidth -g pixelHeight "${imgPath}"`).toString();
    const w   = parseInt(out.match(/pixelWidth:\s*(\d+)/)[1]);
    const h   = parseInt(out.match(/pixelHeight:\s*(\d+)/)[1]);
    return [w, h];
}

async function build() {
    // 找到第一张存在的 cell 用于获取尺寸
    let refImg = null;
    for (const d of DISPLACEMENTS) {
        const p = path.join(CELLS_DIR, `${d}_gt.png`);
        if (fs.existsSync(p)) { refImg = p; break; }
    }
    if (!refImg) {
        console.error('未找到 PNG cells，请先运行 compare_heatmaps.py（PPTX_MODE=True）');
        process.exit(1);
    }

    const [pxW, pxH] = getImgDims(refImg);
    const aspect     = pxW / pxH;
    console.log(`参考图: ${path.basename(refImg)}  (${pxW} × ${pxH} px, aspect=${aspect.toFixed(3)})`);

    // ── 布局参数（单位: 英寸，LAYOUT_16x9 = 10 × 5.625） ────────────────────
    const SW = 10.0, SH = 5.625;
    const MARGIN = 0.08;   // 四周边距
    const LBLW   = 0.55;   // 左侧行标签列宽
    const HDRY   = 0.28;   // 顶部列标题行高
    const GAP_C  = 0.02;   // 列间距
    const GAP_R  = 0.04;   // 行间距

    const nC    = DISPLACEMENTS.length;
    const nR    = ROWS.length;
    const cellW = (SW - 2*MARGIN - LBLW - GAP_C*(nC-1)) / nC;
    const cellH = (SH - 2*MARGIN - HDRY - GAP_R*(nR-1)) / nR;

    // 图片在 cell 内按宽度缩放，保持宽高比
    const imgW = cellW;
    const imgH = imgW / aspect;

    console.log(`每格尺寸: ${cellW.toFixed(3)}″ × ${cellH.toFixed(3)}″`);
    console.log(`图片尺寸: ${imgW.toFixed(3)}″ × ${imgH.toFixed(3)}″`);

    // ── 创建 PPTX ────────────────────────────────────────────────────────────
    const pptx  = new PptxGenJS();
    pptx.layout = 'LAYOUT_16x9';
    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };

    // ── 列标题（1 mm … 8 mm） ────────────────────────────────────────────────
    DISPLACEMENTS.forEach((d, ci) => {
        const x = MARGIN + LBLW + ci * (cellW + GAP_C);
        slide.addText(`${d} mm`, {
            x, y: MARGIN, w: cellW, h: HDRY,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 11, bold: true,
            color: '000000',
        });
    });

    // ── 行标签 + 图片 ─────────────────────────────────────────────────────────
    ROWS.forEach(({ name, label }, ri) => {
        const rowY   = MARGIN + HDRY + ri * (cellH + GAP_R);
        const rowMid = rowY + cellH / 2;

        // 行标签（Times New Roman, 居中）
        slide.addText(label, {
            x: MARGIN, y: rowMid - 0.14, w: LBLW, h: 0.28,
            align: 'center', valign: 'middle',
            fontFace: 'Times New Roman', fontSize: 8, bold: true,
            color: '000000',
        });

        // 各列图片
        DISPLACEMENTS.forEach((d, ci) => {
            const imgPath = path.join(CELLS_DIR, `${d}_${name}.png`);
            if (!fs.existsSync(imgPath)) {
                console.warn(`  跳过（文件不存在）: ${imgPath}`);
                return;
            }
            const x = MARGIN + LBLW + ci * (cellW + GAP_C);
            const y = rowY + (cellH - imgH) / 2;   // 在格内垂直居中
            slide.addImage({ path: imgPath, x, y, w: imgW, h: imgH });
        });
    });

    await pptx.writeFile({ fileName: OUTPUT_PATH });
    console.log(`\n✓ PPTX 已保存: ${OUTPUT_PATH}`);
}

build().catch(err => { console.error(err); process.exit(1); });
