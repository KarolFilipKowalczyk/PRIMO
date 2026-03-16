%% primo.cls — Document class for the PRIMO research programme
%% Karol Filip Kowalczyk, 2026
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{primo}[2026/03/16 PRIMO paper class]

\LoadClass[11pt,a4paper]{article}

%% ── Geometry ──
\RequirePackage[margin=1in]{geometry}

%% ── Spacing ──
\RequirePackage{setspace}
\setstretch{1.15}

%% ── Fonts & math ──
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{mathtools}

%% ── Tables ──
\RequirePackage{booktabs}
\RequirePackage{array}

%% ── Graphics ──
\RequirePackage{graphicx}
\RequirePackage{xcolor}

%% ── Links ──
\RequirePackage[colorlinks=true,linkcolor=blue!60!black,citecolor=green!50!black,urlcolor=blue!70!black]{hyperref}
\RequirePackage[capitalise,noabbrev]{cleveref}

%% ── Headers ──
\RequirePackage{fancyhdr}

%% Short title for header
\newcommand{\primo@shorttitle}{}
\newcommand{\shorttitle}[1]{\renewcommand{\primo@shorttitle}{#1}}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{\primo@shorttitle}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{14pt}

%% First page style: footer with programme info
\fancypagestyle{firstpage}{%
  \fancyhf{}
  \fancyhead[L]{\small\textit{\primo@shorttitle}}
  \fancyhead[R]{\small\thepage}
  \fancyfoot[C]{\small\textit{Part of the PRIMO programme. Repository: \url{https://github.com/KarolFilipKowalczyk/PRIMO}}}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0.4pt}
}

%% ── Author block ──
\newcommand{\primo@affiliation}{}
\newcommand{\primo@email}{}
\newcommand{\affiliation}[1]{\renewcommand{\primo@affiliation}{#1}}
\newcommand{\email}[1]{\renewcommand{\primo@email}{#1}}

\renewcommand{\maketitle}{%
  \thispagestyle{firstpage}%
  \begin{center}
    {\LARGE\bfseries\@title\par}
    \vspace{1em}
    {\large\@author\par}
    \vspace{0.3em}
    {\primo@affiliation\par}
    \vspace{0.2em}
    {\texttt{\primo@email}\par}
    \vspace{0.3em}
    {\@date\par}
  \end{center}
  \vspace{1em}
}

%% ── Abstract ──
\renewenvironment{abstract}{%
  \vspace{0.5em}
  \noindent\textbf{Abstract.}\quad\begin{quote}
}{%
  \end{quote}\vspace{1em}
}

%% ── Theorem environments ──
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{conjecture}[theorem]{Conjecture}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

%% ── Acknowledgments ──
\newenvironment{acknowledgments}{%
  \section*{Acknowledgments}
}{%
  \par
}

%% ── Ragged bottom ──
\raggedbottom

\endinput
