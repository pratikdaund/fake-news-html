// ==============================================================
// Veritas - Fake News Detector (vanilla JS)
// ==============================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------- Element refs ----------
const textarea = $('#article-text');
const charCount = $('#char-count');
const errorBox = $('#error-box');
const loadingBox = $('#loading-box');
const resultBox = $('#result-box');
const analyseBtn = $('#analyse-btn');
const clearBtn = $('#clear-btn');
const modelLabels = $$('.model-toggle label');

// ---------- Sample articles ----------
const SAMPLE_REAL = "The United Nations Security Council on Tuesday unanimously adopted " +
  "a resolution calling for a cessation of hostilities during the upcoming holiday " +
  "period. The 15-member body passed the measure after weeks of negotiations between " +
  "permanent members. The resolution urges all parties to allow humanitarian aid to " +
  "reach civilians in affected regions and calls for the release of hostages taken " +
  "during recent escalations. Several member states expressed cautious optimism that " +
  "the resolution would lead to a de-escalation, while acknowledging significant " +
  "challenges remain in its implementation on the ground.";

const SAMPLE_FAKE = "SHOCKING: Scientists have discovered that drinking lemon water " +
  "every morning can cure all forms of cancer in just 7 days! Big Pharma doesn't " +
  "want you to know this ONE WEIRD TRICK that doctors HATE. A secret study " +
  "suppressed by the government shows that the acidic properties of citrus destroy " +
  "tumors overnight. Thousands have already been cured using this simple method " +
  "that the mainstream media refuses to cover. Share this before it gets deleted! " +
  "The truth they don't want you to see is finally coming out.";

// ---------- Char/word counter ----------
function updateCharCount() {
  const text = textarea.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  charCount.textContent = `${chars} characters · ${words} words`;
}

textarea.addEventListener('input', updateCharCount);

// ---------- Model toggle ----------
modelLabels.forEach((label) => {
  label.addEventListener('click', () => {
    modelLabels.forEach((l) => l.classList.remove('active'));
    label.classList.add('active');
    const input = label.querySelector('input');
    if (input) input.checked = true;
  });
});

function selectedModel() {
  const checked = document.querySelector('input[name="model"]:checked');
  return checked ? checked.value : 'logreg';
}

// ---------- Sample buttons ----------
$('#load-real')?.addEventListener('click', () => {
  textarea.value = SAMPLE_REAL;
  updateCharCount();
  hide(errorBox);
});

$('#load-fake')?.addEventListener('click', () => {
  textarea.value = SAMPLE_FAKE;
  updateCharCount();
  hide(errorBox);
});

// ---------- Clear ----------
clearBtn.addEventListener('click', () => {
  textarea.value = '';
  updateCharCount();
  hide(errorBox);
  hide(resultBox);
});

// ---------- Helpers ----------
function hide(el) { el.classList.add('hidden'); }
function show(el) { el.classList.remove('hidden'); }

function showError(msg) {
  errorBox.textContent = msg;
  show(errorBox);
  hide(loadingBox);
  hide(resultBox);
}

// ---------- Analyse ----------
analyseBtn.addEventListener('click', async () => {
  const text = textarea.value.trim();
  hide(errorBox);
  hide(resultBox);

  if (text.length < 20) {
    showError('Please enter at least 20 characters of text.');
    return;
  }

  show(loadingBox);
  analyseBtn.disabled = true;

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        model: selectedModel(),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || `Request failed: ${response.status}`);
      return;
    }

    renderResult(data);
  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    hide(loadingBox);
    analyseBtn.disabled = false;
  }
});

// ---------- Render result ----------
function renderResult(data) {
  const { label, confidence, model, tokens, real_prob, cleaned } = data;
  const pct = (confidence * 100).toFixed(1);

  // Apply verdict theming
  resultBox.classList.remove('result-real', 'result-fake');
  if (label === 'REAL') {
    resultBox.classList.add('result-real');
  } else if (label === 'FAKE') {
    resultBox.classList.add('result-fake');
  }

  $('#result-label').textContent = label;
  $('#result-pct').textContent = `${pct}%`;
  $('#result-model').textContent = model;
  $('#result-tokens').textContent = tokens;

  // Confidence level
  let level = 'Low confidence — review';
  if (confidence >= 0.9) level = 'High confidence';
  else if (confidence >= 0.7) level = 'Moderate';
  $('#result-status').textContent = level;

  // Animate the bar
  const barFill = $('#bar-fill');
  barFill.style.width = '0%';
  setTimeout(() => { barFill.style.width = `${pct}%`; }, 50);

  // Probability breakdown
  $('#p-real').textContent = real_prob.toFixed(4);
  $('#p-fake').textContent = (1 - real_prob).toFixed(4);

  // Cleaned text
  const cleanedDisplay = cleaned.length > 1000
    ? cleaned.slice(0, 1000) + '...'
    : cleaned;
  $('#cleaned-text').textContent = cleanedDisplay || '(empty)';

  // Low-confidence warning
  if (confidence < 0.7) {
    show($('#low-conf-warning'));
  } else {
    hide($('#low-conf-warning'));
  }

  show(resultBox);
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Init ----------
updateCharCount();

// Keyboard shortcut: Cmd/Ctrl + Enter to submit
textarea.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    analyseBtn.click();
  }
});
