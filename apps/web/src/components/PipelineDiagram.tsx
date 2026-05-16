/**
 * PipelineDiagram — modern React visualization of a fitted sklearn Pipeline.
 *
 * Replaces sklearn's stock HTML repr with a proper interactive diagram:
 *
 *   - Top-level Pipeline renders as a vertical flow with subtle connector lines.
 *   - ColumnTransformer renders as parallel branches in a side-by-side grid,
 *     each branch tagged with the columns it operates on.
 *   - FeatureUnion renders the same way (transformers concatenated horizontally).
 *   - Leaf estimators / transformers render as compact cards with a category
 *     icon, class name, module hint, and click-to-expand params accordion.
 *
 * Each step is colored by its inferred category (imputer / scaler / encoder /
 * transformer / estimator / other) so the data flow is visually scannable.
 *
 * The component is purely presentational — feed it the recursive tree returned
 * by `GET /runs/:id/trials/:tid` (`pipeline_tree`) and it renders the whole
 * thing. Memoised; recursion is shallow in practice (depth ≤ 3 for realistic
 * sklearn pipelines).
 */

import { useState, type ReactNode } from 'react';
import type {
  PipelineNode,
  PipelineColumnTransformerNode,
  PipelineCompositeNode,
  PipelineFeatureUnionNode,
  PipelineLeafNode,
} from '@/api/types';

// ─── Category inference ──────────────────────────────────────────────

type StepCategory =
  | 'imputer'
  | 'scaler'
  | 'encoder'
  | 'transformer'
  | 'selector'
  | 'estimator'
  | 'compose'
  | 'passthrough'
  | 'other';

function inferCategory(node: PipelineNode): StepCategory {
  if (node.type === 'passthrough') return 'passthrough';
  if (node.type === 'pipeline') return 'compose';
  if (node.type === 'column_transformer' || node.type === 'feature_union')
    return 'compose';
  if (node.is_estimator) return 'estimator';
  const cls = node.class.toLowerCase();
  if (cls.includes('imputer') || cls.includes('missing')) return 'imputer';
  if (cls.includes('scaler') || cls.includes('normalizer') || cls.includes('standard'))
    return 'scaler';
  if (cls.includes('encoder') || cls.includes('hasher') || cls.includes('binarizer'))
    return 'encoder';
  if (cls.includes('selector') || cls.includes('selectk') || cls.includes('rfe'))
    return 'selector';
  if (
    cls.includes('transformer') ||
    cls.includes('pca') ||
    cls.includes('poly') ||
    cls.includes('powertransform') ||
    cls.includes('quantile') ||
    cls.includes('discretiz')
  )
    return 'transformer';
  return 'other';
}

const CATEGORY_STYLE: Record<
  StepCategory,
  { ring: string; chip: string; iconBg: string; iconFg: string; label: string }
> = {
  imputer: {
    ring: 'border-amber-300 dark:border-amber-500/40',
    chip: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    iconBg: 'bg-amber-100 dark:bg-amber-500/15',
    iconFg: 'text-amber-700 dark:text-amber-300',
    label: 'Imputer',
  },
  scaler: {
    ring: 'border-sky-300 dark:border-sky-500/40',
    chip: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
    iconBg: 'bg-sky-100 dark:bg-sky-500/15',
    iconFg: 'text-sky-700 dark:text-sky-300',
    label: 'Scaler',
  },
  encoder: {
    ring: 'border-violet-300 dark:border-violet-500/40',
    chip: 'bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300',
    iconBg: 'bg-violet-100 dark:bg-violet-500/15',
    iconFg: 'text-violet-700 dark:text-violet-300',
    label: 'Encoder',
  },
  transformer: {
    ring: 'border-cyan-300 dark:border-cyan-500/40',
    chip: 'bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300',
    iconBg: 'bg-cyan-100 dark:bg-cyan-500/15',
    iconFg: 'text-cyan-700 dark:text-cyan-300',
    label: 'Transformer',
  },
  selector: {
    ring: 'border-fuchsia-300 dark:border-fuchsia-500/40',
    chip: 'bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-500/15 dark:text-fuchsia-300',
    iconBg: 'bg-fuchsia-100 dark:bg-fuchsia-500/15',
    iconFg: 'text-fuchsia-700 dark:text-fuchsia-300',
    label: 'Selector',
  },
  estimator: {
    ring: 'border-accent-400 dark:border-accent-500/50 ring-1 ring-accent-400/20 dark:ring-accent-500/15',
    chip: 'bg-accent-50 text-accent-700 dark:bg-accent-500/15 dark:text-accent-300',
    iconBg: 'bg-accent-500',
    iconFg: 'text-white',
    label: 'Estimator',
  },
  compose: {
    ring: 'border-ink-300 dark:border-ink-700',
    chip: 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-300',
    iconBg: 'bg-ink-100 dark:bg-ink-800',
    iconFg: 'text-ink-700 dark:text-ink-300',
    label: 'Composite',
  },
  passthrough: {
    ring: 'border-dashed border-ink-300 dark:border-ink-700',
    chip: 'bg-ink-50 text-ink-500 dark:bg-ink-900 dark:text-ink-500',
    iconBg: 'bg-ink-100 dark:bg-ink-800',
    iconFg: 'text-ink-500',
    label: 'Passthrough',
  },
  other: {
    ring: 'border-ink-200 dark:border-ink-800',
    chip: 'bg-ink-100 text-ink-600 dark:bg-ink-800 dark:text-ink-300',
    iconBg: 'bg-ink-100 dark:bg-ink-800',
    iconFg: 'text-ink-700 dark:text-ink-300',
    label: 'Step',
  },
};

// ─── Icons ───────────────────────────────────────────────────────────

function CategoryIcon({ cat, className }: { cat: StepCategory; className?: string }) {
  const common = {
    width: 14,
    height: 14,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 2,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    className,
  };
  switch (cat) {
    case 'imputer':
      // sparkle / fix
      return (
        <svg {...common} aria-hidden>
          <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
        </svg>
      );
    case 'scaler':
      // up/down arrows = rescale
      return (
        <svg {...common} aria-hidden>
          <path d="M7 3v18M4 6l3-3 3 3M4 18l3 3 3-3M14 8h6M14 12h6M14 16h6" />
        </svg>
      );
    case 'encoder':
      // hash = categorical encode
      return (
        <svg {...common} aria-hidden>
          <path d="M4 9h16M4 15h16M10 3L8 21M16 3l-2 18" />
        </svg>
      );
    case 'transformer':
      // shuffle
      return (
        <svg {...common} aria-hidden>
          <polyline points="16 3 21 3 21 8" />
          <line x1="4" y1="20" x2="21" y2="3" />
          <polyline points="21 16 21 21 16 21" />
          <line x1="15" y1="15" x2="21" y2="21" />
          <line x1="4" y1="4" x2="9" y2="9" />
        </svg>
      );
    case 'selector':
      // filter funnel
      return (
        <svg {...common} aria-hidden>
          <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
        </svg>
      );
    case 'estimator':
      // crosshair / target
      return (
        <svg {...common} aria-hidden>
          <circle cx="12" cy="12" r="9" />
          <circle cx="12" cy="12" r="5" />
          <circle cx="12" cy="12" r="1.5" fill="currentColor" />
        </svg>
      );
    case 'compose':
      // layers
      return (
        <svg {...common} aria-hidden>
          <polygon points="12 2 2 7 12 12 22 7 12 2" />
          <polyline points="2 17 12 22 22 17" />
          <polyline points="2 12 12 17 22 12" />
        </svg>
      );
    case 'passthrough':
      return (
        <svg {...common} aria-hidden>
          <line x1="5" y1="12" x2="19" y2="12" />
          <polyline points="13 6 19 12 13 18" />
        </svg>
      );
    default:
      return (
        <svg {...common} aria-hidden>
          <rect x="4" y="4" width="16" height="16" rx="3" />
        </svg>
      );
  }
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`transition-transform ${open ? 'rotate-180' : ''}`}
      aria-hidden
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────

function shortModule(mod: string): string {
  if (!mod) return '';
  const parts = mod.split('.');
  if (parts.length <= 2) return mod;
  return `${parts[0]}.${parts[1]}`;
}

function formatParam(v: unknown): string {
  if (v == null) return 'None';
  if (typeof v === 'string') return v || "''";
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  try {
    const json = JSON.stringify(v);
    return json.length > 90 ? `${json.slice(0, 87)}…` : json;
  } catch {
    return String(v);
  }
}

// ─── Main entry ──────────────────────────────────────────────────────

export function PipelineDiagram({ tree }: { tree: PipelineNode | null }) {
  if (!tree) {
    return (
      <div className="card text-sm text-ink-500">
        Pipeline diagram unavailable — the trial pickle could not be inspected.
      </div>
    );
  }
  // Wrap the whole flow with explicit Input + Output endpoints so the
  // DAG has a clear source and sink. Real edges (SVG arrows) connect
  // the body — sklearn pipelines are DAGs at heart and we render them
  // as such instead of a plain vertical accordion.
  return (
    <div className="rounded-xl border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-4 sm:p-6">
      <Endpoint kind="input" />
      <EdgeArrow />
      <NodeRenderer node={tree} depth={0} />
      <EdgeArrow />
      <Endpoint kind="output" />
    </div>
  );
}

function NodeRenderer({ node, depth }: { node: PipelineNode; depth: number }) {
  switch (node.type) {
    case 'pipeline':
      return <PipelineFlow node={node} depth={depth} />;
    case 'column_transformer':
      return <ColumnTransformerSplit node={node} depth={depth} />;
    case 'feature_union':
      return <FeatureUnionSplit node={node} depth={depth} />;
    case 'leaf':
    case 'passthrough':
      return <LeafCard node={node} depth={depth} />;
  }
}

// ─── Pipeline (vertical flow) ───────────────────────────────────────

function PipelineFlow({
  node,
  depth,
}: {
  node: PipelineCompositeNode;
  depth: number;
}) {
  const showWrapper = !node.is_root && depth > 0;
  const content = (
    <div className="flex flex-col">
      {node.children.map((child, idx) => (
        <div key={`${idx}-${child.name}`}>
          <NodeRenderer node={child} depth={depth + 1} />
          {idx < node.children.length - 1 && <EdgeArrow />}
        </div>
      ))}
    </div>
  );
  if (!showWrapper) return content;
  return (
    <CompositeFrame label="Pipeline" name={node.name}>
      {content}
    </CompositeFrame>
  );
}

/**
 * Endpoint pill — `Input` (top) / `Output` (bottom). Same visual weight
 * as a leaf card so it grounds the DAG without dominating it.
 */
function Endpoint({ kind }: { kind: 'input' | 'output' }) {
  const label = kind === 'input' ? 'Input data' : 'Predictions';
  const icon = kind === 'input' ? 'flask' : 'estimator';
  return (
    <div className="flex justify-center">
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-ink-900 dark:bg-white text-white dark:text-ink-900 text-xs font-medium tracking-wide">
        <CategoryIcon cat={icon === 'flask' ? 'compose' : 'estimator'} />
        <span>{label}</span>
      </div>
    </div>
  );
}

/**
 * Vertical edge arrow connecting two stacked DAG nodes. Renders as an
 * SVG so it scales cleanly and the arrowhead actually looks like one.
 */
function EdgeArrow() {
  return (
    <div aria-hidden className="flex justify-center py-1.5">
      <svg
        width="16"
        height="22"
        viewBox="0 0 16 22"
        className="text-ink-300 dark:text-ink-700"
      >
        <line x1="8" y1="0" x2="8" y2="16" stroke="currentColor" strokeWidth="1.5" />
        <polyline
          points="3 14 8 20 13 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

/**
 * Small fixed-aspect arrowhead SVG. Used by the fork/join buses below.
 * Kept in its own component because the buses themselves render lines
 * as plain HTML divs (which don't deform when their container stretches)
 * — only the arrowhead really needs to be SVG.
 */
function ArrowheadDown({ style }: { style: React.CSSProperties }) {
  return (
    <svg
      className="absolute text-ink-300 dark:text-ink-700"
      width="10"
      height="6"
      viewBox="0 0 10 6"
      style={style}
      aria-hidden
    >
      <polyline
        points="1,1 5,5 9,1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * Top fork bus — single line coming in, splitting horizontally to N
 * branch start points.
 *
 * Earlier version was a single SVG with ``preserveAspectRatio="none"``
 * stretched horizontally, which deformed the per-branch arrowheads into
 * flat wedges. We now draw the trunk + bus + legs as 1-pixel HTML divs
 * (no deformation) and only the arrowhead is SVG — that's the part where
 * shape actually matters.
 */
function ForkBus({ count }: { count: number }) {
  if (count <= 1) return null;
  const branchCenterPct = (i: number) => (100 / count) * i + 100 / count / 2;
  const stroke = 'bg-ink-300 dark:bg-ink-700';
  return (
    <div aria-hidden className="relative w-full" style={{ height: 28 }}>
      {/* Trunk down to the bus */}
      <div
        className={`absolute left-1/2 -translate-x-1/2 w-px ${stroke}`}
        style={{ top: 0, height: 10 }}
      />
      {/* Horizontal bus — ends aligned to the outermost branch centers */}
      <div
        className={`absolute h-px ${stroke}`}
        style={{
          top: 10,
          left: `${branchCenterPct(0)}%`,
          right: `${100 - branchCenterPct(count - 1)}%`,
        }}
      />
      {/* Per-branch legs + arrowheads at branch column centers */}
      {Array.from({ length: count }, (_, i) => {
        const leftPct = branchCenterPct(i);
        return (
          <div key={i}>
            <div
              className={`absolute w-px ${stroke}`}
              style={{
                left: `calc(${leftPct}% - 0.5px)`,
                top: 10,
                height: 12,
              }}
            />
            <ArrowheadDown
              style={{ left: `calc(${leftPct}% - 5px)`, top: 22 }}
            />
          </div>
        );
      })}
    </div>
  );
}

/**
 * Bottom join bus — N branches converge into a single output line.
 * Mirrors `ForkBus`; same HTML-lines / SVG-arrowhead split.
 */
function JoinBus({ count }: { count: number }) {
  if (count <= 1) return null;
  const branchCenterPct = (i: number) => (100 / count) * i + 100 / count / 2;
  const stroke = 'bg-ink-300 dark:bg-ink-700';
  return (
    <div aria-hidden className="relative w-full" style={{ height: 28 }}>
      {/* Per-branch legs converging into the bus */}
      {Array.from({ length: count }, (_, i) => {
        const leftPct = branchCenterPct(i);
        return (
          <div
            key={i}
            className={`absolute w-px ${stroke}`}
            style={{ left: `calc(${leftPct}% - 0.5px)`, top: 0, height: 12 }}
          />
        );
      })}
      {/* Horizontal bus */}
      <div
        className={`absolute h-px ${stroke}`}
        style={{
          top: 12,
          left: `${branchCenterPct(0)}%`,
          right: `${100 - branchCenterPct(count - 1)}%`,
        }}
      />
      {/* Trunk down into the next node + arrowhead */}
      <div
        className={`absolute left-1/2 -translate-x-1/2 w-px ${stroke}`}
        style={{ top: 12, height: 10 }}
      />
      <ArrowheadDown style={{ left: 'calc(50% - 5px)', top: 22 }} />
    </div>
  );
}

// ─── ColumnTransformer (horizontal split with column tags) ──────────

function ColumnTransformerSplit({
  node,
  depth,
}: {
  node: PipelineColumnTransformerNode;
  depth: number;
}) {
  const count = node.branches.length;
  return (
    <CompositeFrame
      label="ColumnTransformer"
      name={node.name}
      sublabel={`${count} branch${count === 1 ? '' : 'es'}`}
    >
      <ForkBus count={count} />
      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: `repeat(${Math.min(count, 3)}, minmax(0, 1fr))`,
        }}
      >
        {node.branches.map((b, idx) => (
          <div key={`${idx}-${b.name}`} className="space-y-2">
            <BranchHeader name={b.name} columns={b.columns} />
            <NodeRenderer node={b.transformer} depth={depth + 1} />
          </div>
        ))}
      </div>
      <JoinBus count={count} />
    </CompositeFrame>
  );
}

function BranchHeader({
  name,
  columns,
}: {
  name: string;
  columns: (string | number | boolean)[];
}) {
  const [open, setOpen] = useState(false);
  const visible = open ? columns : columns.slice(0, 4);
  const hidden = columns.length - visible.length;
  return (
    <div className="rounded-md bg-ink-50 dark:bg-ink-950/40 px-3 py-2 border border-ink-200 dark:border-ink-800">
      <div className="text-[11px] uppercase tracking-wide text-ink-500 font-medium">
        {name}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {visible.map((c, i) => (
          <span
            key={`${i}-${String(c)}`}
            className="inline-flex items-center px-1.5 py-0.5 rounded bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 text-[11px] font-mono text-ink-700 dark:text-ink-300"
          >
            {String(c)}
          </span>
        ))}
        {hidden > 0 && (
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="inline-flex items-center px-1.5 py-0.5 rounded bg-ink-100 dark:bg-ink-800 text-[11px] text-ink-600 dark:text-ink-300 hover:bg-ink-200 dark:hover:bg-ink-700"
          >
            +{hidden} more
          </button>
        )}
        {open && columns.length > 4 && (
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="inline-flex items-center px-1.5 py-0.5 rounded bg-ink-100 dark:bg-ink-800 text-[11px] text-ink-600 dark:text-ink-300 hover:bg-ink-200 dark:hover:bg-ink-700"
          >
            collapse
          </button>
        )}
      </div>
    </div>
  );
}

// ─── FeatureUnion ──────────────────────────────────────────────────

function FeatureUnionSplit({
  node,
  depth,
}: {
  node: PipelineFeatureUnionNode;
  depth: number;
}) {
  const count = node.branches.length;
  return (
    <CompositeFrame
      label="FeatureUnion"
      name={node.name}
      sublabel={`${count} branch${count === 1 ? '' : 'es'}`}
    >
      <ForkBus count={count} />
      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: `repeat(${Math.min(count, 3)}, minmax(0, 1fr))`,
        }}
      >
        {node.branches.map((b, idx) => (
          <div key={`${idx}-${b.name}`} className="space-y-2">
            <div className="rounded-md bg-ink-50 dark:bg-ink-950/40 px-3 py-2 border border-ink-200 dark:border-ink-800 text-[11px] uppercase tracking-wide text-ink-500 font-medium">
              {b.name}
            </div>
            <NodeRenderer node={b.transformer} depth={depth + 1} />
          </div>
        ))}
      </div>
      <JoinBus count={count} />
    </CompositeFrame>
  );
}

// ─── Leaf ───────────────────────────────────────────────────────────

function LeafCard({
  node,
  depth: _depth,
}: {
  node: PipelineLeafNode;
  depth: number;
}) {
  const cat = inferCategory(node);
  const style = CATEGORY_STYLE[cat];
  const [open, setOpen] = useState(false);
  const paramEntries = Object.entries(node.params || {}).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const isPassthrough = node.type === 'passthrough';

  return (
    <div
      className={`rounded-lg border bg-white dark:bg-ink-900 transition-shadow hover:shadow-sm ${style.ring}`}
    >
      <button
        type="button"
        onClick={() => !isPassthrough && setOpen(!open)}
        disabled={isPassthrough || paramEntries.length === 0}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left disabled:cursor-default"
        aria-expanded={open}
      >
        <div
          className={`flex-shrink-0 h-8 w-8 rounded-md flex items-center justify-center ${style.iconBg} ${style.iconFg}`}
        >
          <CategoryIcon cat={cat} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="font-mono text-[11px] text-ink-500 uppercase tracking-wide">
              {node.name}
            </span>
            <span className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate">
              {node.class}
            </span>
            <span className={`text-[10px] uppercase tracking-wide font-medium px-1.5 py-0.5 rounded ${style.chip}`}>
              {style.label}
            </span>
          </div>
          {node.module && (
            <div className="text-[11px] text-ink-400 font-mono truncate">
              {shortModule(node.module)}
            </div>
          )}
        </div>
        {!isPassthrough && paramEntries.length > 0 && (
          <div className="text-xs text-ink-500 flex-shrink-0 flex items-center gap-2">
            <span>
              {paramEntries.length} param{paramEntries.length === 1 ? '' : 's'}
            </span>
            <ChevronIcon open={open} />
          </div>
        )}
      </button>
      {open && paramEntries.length > 0 && (
        <div className="border-t border-ink-200 dark:border-ink-800 px-3 py-2.5 bg-ink-50/50 dark:bg-ink-950/30">
          <table className="w-full text-xs">
            <tbody>
              {paramEntries.map(([k, v]) => (
                <tr
                  key={k}
                  className="border-b border-ink-200/60 dark:border-ink-800/60 last:border-0"
                >
                  <td className="py-1.5 pr-4 font-mono text-ink-600 dark:text-ink-400 align-top w-1/3">
                    {k}
                  </td>
                  <td className="py-1.5 font-mono text-ink-900 dark:text-ink-50 break-all">
                    {formatParam(v)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─── Composite frame ───────────────────────────────────────────────

function CompositeFrame({
  label,
  name,
  sublabel,
  children,
}: {
  label: string;
  name: string;
  sublabel?: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-ink-300 dark:border-ink-700 bg-ink-50/50 dark:bg-ink-950/20 p-3">
      <div className="flex items-baseline justify-between mb-3 gap-2 flex-wrap">
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className="text-[10px] uppercase tracking-wider font-semibold text-ink-500">
            {label}
          </span>
          <span className="font-mono text-xs text-ink-700 dark:text-ink-300">
            {name}
          </span>
        </div>
        {sublabel && (
          <span className="text-[11px] text-ink-500 font-mono">{sublabel}</span>
        )}
      </div>
      {children}
    </div>
  );
}
