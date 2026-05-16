/**
 * /workspaces/:wsId/lineage — Phase 4 + 12 lineage viewer.
 *
 * Renders the workspace's lineage edges as a force-directed-ish DAG
 * using plain SVG. Nodes auto-layout via a simple level walk (BFS
 * from sources). Hovering an edge reveals the relation label;
 * clicking a node refocuses the graph at that node with a depth=2
 * subgraph.
 *
 * Deliberately stays Plotly-free — the lineage edge count is tiny
 * vs. the platform's other graphs, and SVG keeps the iframe-friendly
 * footprint small.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { lineageApi } from '@/api/endpoints';
import type { LineageEdge } from '@/api/types';

const KIND_COLOR: Record<string, string> = {
  data_source: '#3b82f6',
  dataset: '#0ea5e9',
  experiment: '#a855f7',
  trial: '#ec4899',
  run: '#f59e0b',
  registered_model: '#10b981',
  registered_model_version: '#059669',
  deployment: '#ef4444',
  analysis: '#6366f1',
};

interface NodeKey {
  kind: string;
  id: string;
}

function nodeKeyStr(n: NodeKey): string {
  return `${n.kind}:${n.id}`;
}

interface LayoutedNode extends NodeKey {
  level: number;
  x: number;
  y: number;
}

function layoutGraph(edges: LineageEdge[]): {
  nodes: LayoutedNode[];
  positions: Map<string, { x: number; y: number }>;
} {
  // Collect nodes + their in/out degree so BFS can pick sources.
  const nodes = new Map<string, NodeKey>();
  const adj = new Map<string, string[]>();
  const inDeg = new Map<string, number>();
  for (const e of edges) {
    const s = nodeKeyStr(e.source);
    const t = nodeKeyStr(e.target);
    nodes.set(s, e.source);
    nodes.set(t, e.target);
    adj.set(s, [...(adj.get(s) ?? []), t]);
    inDeg.set(t, (inDeg.get(t) ?? 0) + 1);
    inDeg.set(s, inDeg.get(s) ?? 0);
  }
  // Levels via BFS from sources (zero in-degree).
  const level = new Map<string, number>();
  const queue: string[] = [];
  for (const [k, d] of inDeg.entries()) {
    if (d === 0) {
      level.set(k, 0);
      queue.push(k);
    }
  }
  // Fall back: if no node had inDeg=0 (cyclic), seed with the first.
  if (queue.length === 0 && nodes.size > 0) {
    const k = nodes.keys().next().value as string;
    level.set(k, 0);
    queue.push(k);
  }
  while (queue.length > 0) {
    const k = queue.shift()!;
    for (const t of adj.get(k) ?? []) {
      const newLevel = (level.get(k) ?? 0) + 1;
      if (!level.has(t) || (level.get(t) ?? 0) < newLevel) {
        level.set(t, newLevel);
        queue.push(t);
      }
    }
  }
  // Bucket by level for x positioning.
  const buckets = new Map<number, string[]>();
  for (const [k, lvl] of level.entries()) {
    const arr = buckets.get(lvl) ?? [];
    arr.push(k);
    buckets.set(lvl, arr);
  }
  const positions = new Map<string, { x: number; y: number }>();
  const colWidth = 220;
  const rowHeight = 70;
  const maxLevel = Math.max(0, ...buckets.keys());
  for (const [lvl, ks] of buckets.entries()) {
    ks.sort();
    ks.forEach((k, i) => {
      const x = 40 + lvl * colWidth;
      const y =
        40 +
        i * rowHeight +
        (rowHeight / 2) * (1 - ks.length / (maxLevel + 1)) * 0;
      positions.set(k, { x, y });
    });
  }
  const layouted: LayoutedNode[] = [];
  for (const [k, node] of nodes.entries()) {
    const pos = positions.get(k) ?? { x: 40, y: 40 };
    layouted.push({ ...node, level: level.get(k) ?? 0, ...pos });
  }
  return { nodes: layouted, positions };
}

export function Lineage() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const [focusNode, setFocusNode] = useState<NodeKey | null>(null);

  const graph = useQuery({
    queryKey: [
      'lineage',
      wsId,
      focusNode ? `${focusNode.kind}:${focusNode.id}` : 'all',
    ],
    queryFn: () =>
      lineageApi.forWorkspace(
        wsId,
        focusNode
          ? { node_kind: focusNode.kind, node_id: focusNode.id, depth: 2 }
          : undefined,
      ),
    enabled: !!wsId,
  });

  const { nodes, positions } = useMemo(
    () => layoutGraph(graph.data?.edges ?? []),
    [graph.data],
  );

  const width = Math.max(
    600,
    ...nodes.map((n) => n.x + 200),
  );
  const height = Math.max(
    400,
    ...nodes.map((n) => n.y + 80),
  );

  return (
    <div className="space-y-6">
      <header>
        <BackButton />
        <h1 className="h-page mt-2">Lineage</h1>
        <p className="muted small">
          {focusNode
            ? `Subgraph around ${focusNode.kind}:${focusNode.id.slice(0, 8)}… (depth 2)`
            : 'Every edge across this workspace. Click a node to focus.'}
        </p>
      </header>

      {focusNode && (
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={() => setFocusNode(null)}
        >
          ← clear focus
        </button>
      )}

      {graph.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}

      {graph.data && (graph.data.edges?.length ?? 0) === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No edges yet</h3>
          <p className="mt-1 text-sm text-ink-500">
            Lineage edges are written as Trials are run, models promoted,
            and deployments created.
          </p>
        </div>
      )}

      {graph.data && (graph.data.edges?.length ?? 0) > 0 && (
        <div className="card overflow-auto p-0">
          <svg
            width={width}
            height={height}
            style={{ display: 'block' }}
            role="img"
            aria-label="lineage graph"
          >
            <defs>
              <marker
                id="arrow"
                markerWidth="10"
                markerHeight="10"
                refX="10"
                refY="5"
                orient="auto"
              >
                <path d="M0,0 L10,5 L0,10 Z" fill="#94a3b8" />
              </marker>
            </defs>
            {graph.data.edges.map((e) => {
              const s = positions.get(nodeKeyStr(e.source));
              const t = positions.get(nodeKeyStr(e.target));
              if (!s || !t) return null;
              return (
                <g key={e.id}>
                  <line
                    x1={s.x + 80}
                    y1={s.y + 20}
                    x2={t.x}
                    y2={t.y + 20}
                    stroke="#94a3b8"
                    strokeWidth={1.5}
                    markerEnd="url(#arrow)"
                  />
                  <text
                    x={(s.x + 80 + t.x) / 2}
                    y={(s.y + t.y) / 2 + 15}
                    fontSize="10"
                    fill="#64748b"
                    textAnchor="middle"
                  >
                    {e.relation}
                  </text>
                </g>
              );
            })}
            {nodes.map((n) => (
              <g
                key={nodeKeyStr(n)}
                transform={`translate(${n.x},${n.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setFocusNode({ kind: n.kind, id: n.id })}
              >
                <rect
                  x={0}
                  y={0}
                  width={170}
                  height={40}
                  rx={6}
                  fill={KIND_COLOR[n.kind] ?? '#64748b'}
                  fillOpacity={0.1}
                  stroke={KIND_COLOR[n.kind] ?? '#64748b'}
                />
                <text x={10} y={16} fontSize="10" fill="#475569">
                  {n.kind}
                </text>
                <text x={10} y={30} fontSize="12" fontWeight="600" fill="#0f172a">
                  {n.id.slice(0, 8)}…
                </text>
              </g>
            ))}
          </svg>
        </div>
      )}

      {graph.data && (graph.data.edges?.length ?? 0) > 0 && (
        <section>
          <h2 className="h-section mb-2">Edges</h2>
          <div className="card overflow-hidden p-0">
            <table className="w-full text-xs">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-3 py-1.5 text-left font-medium">Source</th>
                  <th className="px-3 py-1.5 text-left font-medium">Relation</th>
                  <th className="px-3 py-1.5 text-left font-medium">Target</th>
                  <th className="px-3 py-1.5 text-left font-medium">When</th>
                </tr>
              </thead>
              <tbody>
                {graph.data.edges.map((e) => (
                  <tr
                    key={e.id}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-3 py-1.5">
                      <code className="text-xs">
                        {e.source.kind}:{e.source.id.slice(0, 8)}…
                      </code>
                    </td>
                    <td className="px-3 py-1.5">
                      <span className="pill-neutral">{e.relation}</span>
                    </td>
                    <td className="px-3 py-1.5">
                      <code className="text-xs">
                        {e.target.kind}:{e.target.id.slice(0, 8)}…
                      </code>
                    </td>
                    <td className="px-3 py-1.5 text-ink-500">
                      {e.created_at
                        ? new Date(e.created_at).toLocaleString()
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
