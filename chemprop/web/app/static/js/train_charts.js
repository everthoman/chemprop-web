// Shared chart rendering utilities — used by train.html and checkpoints.html

function _ensureStructTooltip() {
    var t = document.getElementById('struct-tooltip');
    if (!t) {
        t = document.createElement('div');
        t.id = 'struct-tooltip';
        t.style.cssText = 'display:none;position:fixed;z-index:9999;background:white;border:1px solid #ccc;' +
            'border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,0.18);padding:8px 10px;pointer-events:none;max-width:450px;';
        document.body.appendChild(t);
    }
    return t;
}

function makeStatsTable(task, datasetType) {
    var table = document.createElement('table');
    table.className = 'table table-sm table-bordered';
    table.style.marginBottom = '8px';
    if (datasetType === 'regression') {
        if (!task.train_stats || !task.test_stats) return null;
        var ts = task.train_stats, xs = task.test_stats;
        table.style.maxWidth = '380px';
        table.innerHTML =
            '<thead><tr><th>Metric</th><th>Train (n=' + ts.n + ')</th><th>Test (n=' + xs.n + ')</th></tr></thead>' +
            '<tbody>' +
            '<tr><td>R² / Q²</td><td>' + ts.r2 + '</td><td>' + xs.r2 + '</td></tr>' +
            '<tr><td>RMSE</td><td>' + ts.rmse + '</td><td>' + xs.rmse + '</td></tr>' +
            '<tr><td>MAE</td><td>' + ts.mae + '</td><td>' + xs.mae + '</td></tr>' +
            '</tbody>';
    } else if (datasetType === 'classification') {
        if (!task.test_stats) return null;
        var xs = task.test_stats;
        var wrapper = document.createElement('div');
        wrapper.style.cssText = 'display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; margin-bottom:8px';
        table.style.maxWidth = '280px';
        table.style.marginBottom = '0';
        table.innerHTML =
            '<thead><tr><th>Metric</th><th>Test (n=' + xs.n + ')</th></tr></thead>' +
            '<tbody>' +
            '<tr><td>Class balance</td><td>' + xs.n_pos + ' pos / ' + xs.n_neg + ' neg (' +
            Math.round(xs.n_pos / xs.n * 1000) / 10 + '% / ' +
            Math.round(xs.n_neg / xs.n * 1000) / 10 + '%)</td></tr>' +
            '<tr><td>AUC</td><td>' + xs.auc + '</td></tr>' +
            '<tr><td>Accuracy</td><td>' + xs.accuracy + '</td></tr>' +
            '<tr><td>Precision</td><td>' + xs.precision + '</td></tr>' +
            '<tr><td>Recall</td><td>' + xs.recall + '</td></tr>' +
            '<tr><td>Specificity</td><td>' + xs.specificity + '</td></tr>' +
            '<tr><td>F1</td><td>' + xs.f1 + '</td></tr>' +
            '<tr><td>MCC</td><td>' + xs.mcc + '</td></tr>' +
            '</tbody>';
        var cm = document.createElement('table');
        cm.className = 'table table-sm table-bordered';
        cm.style.cssText = 'max-width:280px; margin-bottom:0; text-align:center';
        cm.innerHTML =
            '<thead><tr><th></th><th colspan="2">Predicted</th></tr>' +
            '<tr><th></th><th>Negative</th><th>Positive</th></tr></thead>' +
            '<tbody>' +
            '<tr><th style="vertical-align:middle;text-align:right">Actual Negative</th>' +
            '<td style="background:#d4edda">TN: ' + xs.tn + '</td>' +
            '<td style="background:#f8d7da">FP: ' + xs.fp + '</td></tr>' +
            '<tr><th style="vertical-align:middle;text-align:right">Actual Positive</th>' +
            '<td style="background:#f8d7da">FN: ' + xs.fn + '</td>' +
            '<td style="background:#d4edda">TP: ' + xs.tp + '</td></tr>' +
            '</tbody>';
        wrapper.appendChild(table);
        wrapper.appendChild(cm);
        return wrapper;
    } else {
        return null;
    }
    return table;
}

function renderConvergenceChart(canvasId, valCurves) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !valCurves || !valCurves.models || !valCurves.models.length) return null;
    var colors = ['rgba(54,162,235,1)', 'rgba(255,99,132,1)', 'rgba(75,192,192,1)', 'rgba(255,159,64,1)'];
    var datasets = valCurves.models.map(function(scores, i) {
        return {
            label: 'Model ' + i,
            data: scores.map(function(s, j) { return {x: j, y: s}; }),
            borderColor: colors[i % colors.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
            fill: false
        };
    });
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { datasets: datasets },
        options: {
            animation: false,
            plugins: { legend: { display: true, labels: { usePointStyle: true, pointStyle: 'line', pointStyleWidth: 24, lineWidth: 2 } } },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Epoch' } },
                y: { title: { display: true, text: valCurves.metric } }
            }
        }
    });
}

function initCharts(plotData, datasetType, ckptId) {
    var tooltip = _ensureStructTooltip();
    var svgCache = {};
    var hoverTimer = null;
    var lastKey = null;

    function positionTooltip(clientX, clientY) {
        var w = tooltip.offsetWidth || 440;
        var left = clientX + 18;
        if (left + w > window.innerWidth - 10) left = clientX - w - 18;
        var top = clientY - 60;
        if (top < 10) top = 10;
        tooltip.style.left = left + 'px';
        tooltip.style.top  = top  + 'px';
    }

    function showLoading(clientX, clientY, actual, pred, split) {
        tooltip.innerHTML =
            '<div style="font:12px monospace;color:#555;margin-bottom:4px">' +
            (split === 'train' ? 'Train' : 'Test') +
            '  |  Exp: ' + actual.toFixed(3) +
            '  |  Pred: ' + pred.toFixed(3) + '</div>' +
            '<div style="color:#aaa;font-size:12px">Loading structure…</div>';
        tooltip.style.display = 'block';
        positionTooltip(clientX, clientY);
    }

    function showSVG(clientX, clientY, actual, pred, split, smiles, svg, hasAttribution) {
        var legend = hasAttribution
            ? '<div style="font-size:0.72em;color:#666;text-align:center;margin-top:2px">' +
              '<span style="color:#2ecc71">&#9632;</span> increases  ' +
              '<span style="color:#e74c3c">&#9632;</span> decreases prediction</div>'
            : '';
        tooltip.innerHTML =
            '<div style="font:12px monospace;color:#555;margin-bottom:2px">' +
            (split === 'train' ? 'Train' : 'Test') +
            '  |  Exp: ' + actual.toFixed(3) +
            '  |  Pred: ' + pred.toFixed(3) + '</div>' +
            svg + legend;
        tooltip.style.display = 'block';
        positionTooltip(clientX, clientY);
    }

    function hideTooltip() {
        tooltip.style.display = 'none';
        lastKey = null;
        if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
    }

    function onPointHover(event, taskData, dsIdx, ptIdx) {
        var split = dsIdx === 0 ? 'train' : 'test';
        var pts = split === 'train' ? taskData.train : taskData.test;
        if (!pts[ptIdx]) return;
        var actual = pts[ptIdx][0], pred = pts[ptIdx][1], smiles = pts[ptIdx][2];
        if (!smiles) return;
        var cx = event.native.clientX, cy = event.native.clientY;
        if (smiles === lastKey) {
            if (tooltip.style.display === 'block') positionTooltip(cx, cy);
            return;
        }
        lastKey = smiles;
        if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
        if (svgCache[smiles]) {
            showSVG(cx, cy, actual, pred, split, smiles, svgCache[smiles].svg, svgCache[smiles].hasAttribution);
            return;
        }
        showLoading(cx, cy, actual, pred, split);
        hoverTimer = setTimeout(function() {
            fetch('/get_attribution?smiles=' + encodeURIComponent(smiles) + '&ckpt_id=' + encodeURIComponent(ckptId))
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.svg) {
                        svgCache[smiles] = { svg: data.svg, hasAttribution: !!data.has_attribution };
                        if (lastKey === smiles) showSVG(cx, cy, actual, pred, split, smiles, data.svg, !!data.has_attribution);
                    }
                })
                .catch(function(e) { console.warn('Attribution fetch failed:', e); });
        }, 300);
    }

    plotData.forEach(function(task, i) {
        var canvas = document.getElementById('chart_' + i);
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        if (datasetType === 'regression') {
            var allVals = [];
            task.train.concat(task.test).forEach(function(d) { allVals.push(d[0], d[1]); });
            allVals = allVals.filter(function(v) { return typeof v === 'number' && isFinite(v); });
            if (!allVals.length) return;
            var minVal = Math.min.apply(null, allVals);
            var maxVal = Math.max.apply(null, allVals);
            var pad = (maxVal - minVal) * 0.05 || 0.5;
            var axisMin = minVal - pad, axisMax = maxVal + pad;
            new Chart(ctx, {
                type: 'scatter',
                data: { datasets: [
                    { label: 'Train', data: task.train.map(function(d) { return {x: d[0], y: d[1]}; }), backgroundColor: 'rgba(54,162,235,0.4)', pointRadius: 3 },
                    { label: 'Test',  data: task.test.map(function(d)  { return {x: d[0], y: d[1]}; }), backgroundColor: 'rgba(255,99,132,0.7)',  pointRadius: 4 },
                    { label: 'y = x', type: 'line', data: [{x: axisMin, y: axisMin}, {x: axisMax, y: axisMax}], borderColor: 'black', borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, pointStyle: 'line', fill: false }
                ]},
                options: {
                    plugins: { legend: { display: true, labels: { filter: function(item) { return item.text !== 'y = x'; } } } },
                    scales: {
                        x: { type: 'linear', title: { display: true, text: 'Experimental' }, min: axisMin, max: axisMax },
                        y: { type: 'linear', title: { display: true, text: 'Predicted'    }, min: axisMin, max: axisMax }
                    },
                    onHover: function(event, elements) {
                        if (elements.length > 0 && elements[0].datasetIndex < 2) {
                            ctx.canvas.style.cursor = 'pointer';
                            onPointHover(event, task, elements[0].datasetIndex, elements[0].index);
                        } else {
                            ctx.canvas.style.cursor = 'default';
                            hideTooltip();
                        }
                    }
                }
            });
        } else {
            new Chart(ctx, {
                type: 'line',
                data: { datasets: [
                    { label: task.name, data: task.fpr.map(function(f, idx) { return {x: f, y: task.tpr[idx]}; }), borderColor: 'rgba(54,162,235,1)', backgroundColor: 'rgba(54,162,235,0.1)', fill: true, pointRadius: 0, tension: 0 },
                    { label: 'Random', data: [{x:0,y:0},{x:1,y:1}], borderColor: 'rgba(150,150,150,1)', borderDash: [6,4], borderWidth: 2, fill: false, pointRadius: 0, pointStyle: 'line' }
                ]},
                options: {
                    plugins: { legend: { labels: { usePointStyle: true, generateLabels: function(chart) {
                        return Chart.defaults.plugins.legend.labels.generateLabels(chart).map(function(item) {
                            if (item.text === 'Random') { item.lineDash = [6, 4]; item.lineWidth = 2; }
                            return item;
                        });
                    }}}},
                    scales: {
                        x: { type: 'linear', title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                        y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
                    }
                }
            });
        }
    });
}
