"""Convert the working draft's XML export into paper/draft/main.tex.

  python paper/draft/doc_to_tex.py draft.xml paper/draft/main.tex
  cd paper/draft && tectonic -X compile main.tex
"""
import re, sys
import xml.etree.ElementTree as ET

SRC, OUT = sys.argv[1], sys.argv[2]
root = ET.fromstring(open(SRC).read())

CITES = {
    'Adams et al., 2021': 'adams2021uniswap', 'Angeris et al., 2021': 'angeris2021analysis',
    'Bengio et al., 2015': 'bengio2015scheduled', 'Bollerslev, 1986': 'bollerslev1986garch',
    'Byrd et al., 2020': 'byrd2020abides', 'Cont, 2001': 'cont2001stylized',
    'Di Nosse and Lillo, 2026': 'dinosse2026dynamic', 'Durkan et al., 2019': 'durkan2019spline',
    'Ferro, 2014': 'ferro2014fair', 'Gneiting and Raftery, 2007': 'gneiting2007scoring',
    'Hawkes, 1971': 'hawkes1971', 'Lambert et al., 2020': 'lambert2020objective',
    'Loesch et al., 2021': 'loesch2021impermanent', 'Mei and Eisner, 2017': 'mei2017neuralhawkes',
    'Milionis et al., 2022': 'milionis2022lvr', 'Palit, 2022': 'palit2022shuffled',
    'Ranzato et al., 2016': 'ranzato2016sequence', 'Shchur et al., 2020': 'shchur2020intensityfree',
    'Shi and Cartlidge, 2022': 'shi2022sdpnhp', 'Theis et al., 2016': 'theis2016note',
    'Uniswap Governance, 2025': 'uniswapgov2025', 'Uniswap Governance, 2026': 'uniswapgov2026',
    'Venkatraman et al., 2015': 'venkatraman2015improving', 'Wan and Adams, 2022': 'wan2022jit',
    'Wiese et al., 2020': 'wiese2020quantgans', 'Zuo et al., 2020': 'zuo2020transformerhawkes',
}
FIGS = {'1': 'fig1_architecture', '2': 'fig2_forecast_cones', '3': 'fig3_jit',
        '4': 'fig4_stylized_facts', '5': 'fig5_nll_vs_rollout', 'S1': 'figS1_pit'}

# Exact doc substrings rendered as LaTeX math (applied before escaping).
MATH = [
    ('2×10⁻⁵', r'$2\times10^{-5}$'),
    ('learning rate 10^-4', r'learning rate $10^{-4}$'), ('log(L/Lactive)', r'$\log(L/L_{\mathrm{active}})$'),
    ('wᵢ = 0.9·softmaxᵢ(sᵢ) + 0.1/n, with score sᵢ = 2·tanh(cᵀxᵢ/√k).',
     r'\[ w_i = 0.9\,\mathrm{softmax}_i(s_i) + 0.1/n \quad\text{with score}\quad s_i = 2\tanh\bigl(c^{\top} x_i/\sqrt{k}\bigr). \]'),
    ('d ∈ {0, 1, 2, …}', r'$d \in \{0, 1, 2, \dots\}$'), ('where d = 0', r'where $d = 0$'),
    ('gap d ≥ 1', r'gap $d \geq 1$'),
    ('P(d) = Σₖ πₖ (1 − e^(−rₖ)) e^(−(d−1) rₖ)', r'$P(d) = \sum_k \pi_k \bigl(1 - e^{-r_k}\bigr)\, e^{-(d-1) r_k}$'),
    ('mixture weights πₖ and rates rₖ', r'mixture weights $\pi_k$ and rates $r_k$'),
    ('liquidity L over a price range [pa,pb)', r'liquidity $L$ over a price range $[p_a, p_b)$'),
    ('L/√P', r'$L/\sqrt{P}$'), ('√(s₁²/3 + s₂²/3)', r'$\sqrt{s_1^2/3 + s_2^2/3}$'),
    ('Δ energy', r'$\Delta$ energy'), ('1 − 1/e ≈ 63%', r'$1 - 1/e \approx 63\%$'),
    ('moves √P from a to b', r'moves $\sqrt{P}$ from $a$ to $b$'),
    ('f/(1−f)·|b − a|', r'$\frac{f}{1-f}\,|b - a|$'), ('f/(1−f)·|1/b − 1/a|', r'$\frac{f}{1-f}\,|1/b - 1/a|$'),
    ('where f is the fee tier', r'where $f$ is the fee tier'), ('because √P moves', r'because $\sqrt{P}$ moves'),
    ('Each path i in a seed\'s ensemble of n paths', r"Each path $i$ in a seed's ensemble of $n$ paths"),
    ('wᵢ = 0.9·softmaxᵢ(sᵢ) + 0.1/n', r'$w_i = 0.9\,\mathrm{softmax}_i(s_i) + 0.1/n$'),
    ('sᵢ = 2·tanh(cᵀxᵢ/√k)', r'$s_i = 2\tanh\bigl(c^{\top} x_i/\sqrt{k}\bigr)$'),
    ('The k = 25 features xᵢ', r'The $k = 25$ features $x_i$'),
    ('coefficient vector c;', r'coefficient vector $c$;'), ('plus λ times', r'plus $\lambda$ times'),
    ('λ ∈ {0.001, 0.01, 0.1}', r'$\lambda \in \{0.001, 0.01, 0.1\}$'), ('(λ = 0.1 for', r'($\lambda = 0.1$ for'),
    ('E_w|X − y| − ½ Σᵢ≠ⱼ wᵢwⱼ|Xᵢ − Xⱼ| / (1 − Σᵢ wᵢ²)',
     r'\[ \mathbb{E}_w|X - y| - \frac{\sum_{i \neq j} w_i w_j |X_i - X_j|}{2\,(1 - \sum_i w_i^2)} \]'),
    ('log(1+L(offset)/L(now))/4', r'$\log\bigl(1 + L_{\mathrm{offset}}/L_{\mathrm{now}}\bigr)/4$'),
    ('between -4% and +4%', r'between $-4\%$ and $+4\%$'),
]
UNI = {'−': r'\ensuremath{-}', '±': r'\ensuremath{\pm}', '×': r'\ensuremath{\times}', '·': r'\ensuremath{\cdot}',
       '≈': r'\ensuremath{\approx}', 'Δ': r'\ensuremath{\Delta}', 'α': r'\ensuremath{\alpha}', '’': "'", '…': r'\dots{}', '–': '--', '—': '---', 'š': r'\v{s}', 'ü': r'\"u'}
ALLOWED = set()

stash = []
def keep(s):
    stash.append(s); return f'\x00{len(stash) - 1}\x00'

def cite_piece(p):
    m = re.fullmatch(r'(.+?), (\d{4}(?:, \d{4})*)', p.strip())
    if not m: return None
    keys = [CITES.get(f'{m.group(1)}, {y}') for y in m.group(2).split(', ')]
    return keys if all(keys) else None

def citations(s):
    def paren(m):
        pieces = m.group(1).split('; ')
        got = [cite_piece(p) for p in pieces]
        if not any(got): return m.group(0)
        if all(got): return keep(r'\citep{' + ','.join(k for g in got for k in g) + '}')
        out = []
        for p, g in zip(pieces, got): out.append(keep(r'\citealp{' + ','.join(g) + '}') if g else p)
        return '(' + '; '.join(out) + ')'
    s = re.sub(r'\(([^()]*)\)', paren, s)
    for auth, key in CITES.items():
        name, yr = auth.rsplit(', ', 1)
        s = s.replace(f'{name} ({yr})', keep(r'\citet{' + key + '}'))
    return s

def tex(s):
    for a, b in MATH: s = s.replace(a, keep(b))
    s = re.sub(r'https?://\S+?(?=[\s)]|$|\.\s|\.$)', lambda m: keep(r'\url{' + m.group(0) + '}'), s)
    s = citations(s)
    s = s.replace('\\', r'\textbackslash{}')
    for c in '&%$#_{}': s = s.replace(c, '\\' + c)
    s = s.replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}')
    for a, b in UNI.items(): s = s.replace(a, b)
    bad = {c for c in s if ord(c) > 127 and c not in ALLOWED}
    if bad: raise SystemExit(f'unmapped characters {bad} in: {s[:200]}')
    while '\x00' in s: s = re.sub(r'\x00(\d+)\x00', lambda m: stash[int(m.group(1))], s)
    return s

def runs(e, styled=True):
    """Inline content with bold/italic marks as LaTeX."""
    out = []
    for t in e.iter('text'):
        raw = ''.join(t.itertext())
        if not raw: continue
        b, i = t.find('bold') is not None, t.find('italic') is not None
        out.append((raw, b and styled, i and styled))
    # merge adjacent runs with the same style, then convert
    merged = []
    for r in out:
        if merged and merged[-1][1:] == r[1:]: merged[-1] = (merged[-1][0] + r[0],) + r[1:]
        else: merged.append(r)
    s = ''
    for raw, b, i in merged:
        x = tex(raw)
        if b and x.strip(): x = r'\textbf{' + x + '}'
        elif i and x.strip(): x = r'\emph{' + x + '}'
        s += x
    return s

def text(e): return ''.join(e.itertext())

def lead(e):
    """A paragraph opening with a bold label 'Label.' becomes \\paragraph{Label.}."""
    ts = [t for t in e.iter('text') if ''.join(t.itertext())]
    if ts and ts[0].find('bold') is not None:
        label = ''
        while ts and ts[0].find('bold') is not None: label += ''.join(ts.pop(0).itertext())
        if label.strip().endswith(('.', ':')):
            rest = ET.Element('p'); rest.extend(ts)
            return label.strip(), runs(rest)
    return None, runs(e)

def caption(e):
    s = text(e); m = re.match(r'(Table|Figure)\s+(S?\d+):\s*', s)
    body = ET.Element('p')
    for t in e.iter('text'): body.append(t)
    c = runs(body, styled=False)
    return m.group(1), m.group(2), re.sub(r'^(Table|Figure)\s+S?\d+:\s*', '', c)

def numeric(s): return bool(re.fullmatch(r'[\s\d.,%×±+\-−\[\]/]+', s)) and any(ch.isdigit() for ch in s)

def table(e, cap, label):
    rows = [[runs(c) for c in r] for r in e.findall('row')]
    raw = [[text(c) for c in r] for r in e.findall('row')]
    n = len(rows[0])
    align = ''.join('r' if sum(numeric(r[j]) for r in raw[1:]) > len(raw[1:]) / 2 else 'l' for j in range(n))
    group = raw[0][0] == 'Period'
    L = [r'\begin{table}[t]', r'\centering', r'\caption{' + cap + r'}\label{tab:' + label + '}',
         r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{' + align + '}', r'\toprule',
         ' & '.join(rows[0]) + r' \\', r'\midrule']
    prev = None
    for r, rr in zip(rows[1:], raw[1:]):
        key = tuple(rr[:2])
        if group and prev is not None and key != prev: L.append(r'\midrule')
        prev = key
        L.append(' & '.join(r) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    return '\n'.join(L)

def figure(num, cap):
    return '\n'.join([r'\begin{figure}[t]', r'\centering', r'\includegraphics[width=\linewidth]{' + FIGS[num] + '.pdf}',
                      r'\caption{' + cap + r'}\label{fig:' + num + '}', r'\end{figure}'])

blocks = list(root)
title = text(blocks[1]).strip()
body, i, abstract = [], 0, ''
pending_caption = None
while i < len(blocks):
    e = blocks[i]; i += 1
    h = e.get('heading'); s = text(e).strip()
    if e.tag == 'paragraph' and (i - 1) <= 2: continue
    if e.tag == 'paragraph' and not s: continue
    if h == '2':
        if s == 'Abstract': abstract = runs(blocks[i]); i += 1; continue
        if s == 'References': break
        if s.startswith('Appendix'):
            body += [r'\appendix', r'\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\theHfigure}{S\arabic{figure}}\setcounter{figure}{0}',
                     r'\section{' + tex(s.split(':', 1)[1].strip()) + '}']
            continue
        body.append(r'\section{' + tex(re.sub(r'^\d+\s+', '', s)) + '}'); continue
    if h == '3':
        name = re.sub(r'^\d+(\.\d+)*\s+', '', s)
        body.append((r'\subsection{' + tex(name) + '}') if name != s else (r'\paragraph{' + tex(name) + '.}'))
        continue
    if e.tag == 'paragraph' and re.match(r'(Table|Figure)\s+S?\d+:', s):
        kind, num, cap = caption(e)
        if kind == 'Figure': body.append(figure(num, cap))
        else: pending_caption = (num, cap)
        continue
    if e.tag == 'table':
        num, cap = pending_caption; pending_caption = None
        body.append(table(e, cap, num)); continue
    if e.tag == 'list':
        env = 'enumerate' if e.get('kind') == 'ordered' else 'itemize'
        items = []
        for li in e.findall('listItem'):
            label, rest = lead(li)
            items.append(r'\item ' + (r'\textbf{' + tex(label) + '} ' if label else '') + rest.strip())
        body.append('\\begin{%s}\n%s\n\\end{%s}' % (env, '\n'.join(items), env)); continue
    if e.tag == 'paragraph':
        label, rest = lead(e)
        body.append((r'\paragraph{' + tex(label) + '} ' if label else '') + rest.strip()); continue
    raise SystemExit(f'unhandled block {e.tag} {e.get("id")}')

head = r"""% DEX-Net paper. Generated from the working draft (Claude Docs); edit the doc, then regenerate.
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage[round]{natbib}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\graphicspath{{../figures/}}

\title{""" + tex(title).replace('DEX-Net: ', r'DEX-Net:\\ ') + r"""}
\author{Clayton Barnes}
\date{}

\begin{document}
\maketitle

\begin{abstract}
""" + abstract + r"""
\end{abstract}

"""
tail = r"""

\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
"""
# Appendix must come after the bibliography? Keep appendix before references as in the doc.
open(OUT, 'w').write(head + '\n\n'.join(body) + tail)
print('wrote', OUT, len(body), 'blocks')
