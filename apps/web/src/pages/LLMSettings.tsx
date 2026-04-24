/**
 * /workspaces/:wsId/llm — configure the workspace's LLM provider.
 *
 * Per DECISIONS.md § session-13 · decision-3, the backend router supports
 * Anthropic (Claude) and OpenAI as first-class providers from day one. The
 * form mirrors that: pick provider, enter API key (plaintext — v1), model
 * name, optional base_url. Switching providers keeps the prior row for
 * audit but flips its ``enabled`` flag off server-side.
 */

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { llmApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { LLMProviderName } from '@/api/types';

type ProviderChoice = {
  value: LLMProviderName;
  label: string;
  defaultModel: string;
  supported: boolean;
};

const PROVIDERS: ProviderChoice[] = [
  { value: 'anthropic', label: 'Anthropic (Claude)', defaultModel: 'claude-sonnet-4-5', supported: true },
  { value: 'openai', label: 'OpenAI (GPT)', defaultModel: 'gpt-4o-mini', supported: true },
  { value: 'google', label: 'Google Gemini', defaultModel: '', supported: false },
  { value: 'azure_openai', label: 'Azure OpenAI', defaultModel: '', supported: false },
  { value: 'ollama', label: 'Ollama (local)', defaultModel: '', supported: false },
  { value: 'custom_openai_compatible', label: 'Custom OpenAI-compatible', defaultModel: '', supported: false },
];

export function LLMSettings() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const settings = useQuery({
    queryKey: ['llm', 'settings', wsId],
    queryFn: () => llmApi.getSettings(wsId),
    enabled: !!wsId,
  });

  // Form state. Seeds from loaded settings on first fetch.
  const [provider, setProvider] = useState<LLMProviderName>('anthropic');
  const [apiKey, setApiKey] = useState('');
  const [modelName, setModelName] = useState('claude-sonnet-4-5');
  const [baseUrl, setBaseUrl] = useState('');
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    if (!settings.data) return;
    setProvider(settings.data.provider);
    setModelName(settings.data.model_name);
    setBaseUrl(settings.data.base_url ?? '');
    setEnabled(settings.data.enabled);
  }, [settings.data]);

  const save = useMutation({
    mutationFn: () =>
      llmApi.upsertSettings(wsId, {
        provider,
        model_name: modelName.trim(),
        api_key: apiKey.trim() || null,
        base_url: baseUrl.trim() || null,
        enabled,
      }),
    onSuccess: () => {
      setApiKey(''); // clear the textbox — we don't round-trip the plaintext
      qc.invalidateQueries({ queryKey: ['llm', 'settings', wsId] });
    },
  });

  const test = useMutation({
    mutationFn: () => llmApi.testConnection(wsId),
  });

  const providerSupported =
    PROVIDERS.find((p) => p.value === provider)?.supported ?? false;
  const canSave =
    providerSupported &&
    modelName.trim().length > 0 &&
    (settings.data?.has_api_key || apiKey.trim().length > 0) &&
    !save.isPending;

  return (
    <div className="space-y-8 max-w-2xl">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-100">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>LLM</span>
        </nav>
        <h1 className="text-xl font-semibold">LLM provider settings</h1>
        <p className="mt-1 text-sm text-ink-200/70">
          Configure which LLM answers dataset + experiment consultations for this
          workspace. The LLM is <strong>advisory</strong>: it proposes configs +
          risk flags, and the deterministic engine executes what you approve.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (canSave) save.mutate();
        }}
        className="card space-y-5"
      >
        <div>
          <label className="field" htmlFor="provider">
            Provider <span className="text-danger-500">*</span>
          </label>
          <select
            id="provider"
            className="input"
            value={provider}
            onChange={(e) => {
              const next = e.target.value as LLMProviderName;
              setProvider(next);
              const def = PROVIDERS.find((p) => p.value === next)?.defaultModel;
              if (def) setModelName(def);
            }}
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value} disabled={!p.supported}>
                {p.label}
                {!p.supported ? ' — (coming later)' : ''}
              </option>
            ))}
          </select>
          <p className="hint mt-1">
            Anthropic and OpenAI are first-class today. More providers hit this
            same router pattern; see <code className="font-mono">DECISIONS.md</code>.
          </p>
        </div>

        <div>
          <label className="field" htmlFor="model">
            Model name <span className="text-danger-500">*</span>
          </label>
          <input
            id="model"
            className="input"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder={provider === 'anthropic' ? 'claude-sonnet-4-5' : 'gpt-4o-mini'}
            required
          />
        </div>

        <div>
          <label className="field" htmlFor="apikey">
            API key {settings.data?.has_api_key ? '(stored)' : <span className="text-danger-500">*</span>}
          </label>
          <input
            id="apikey"
            className="input font-mono"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={settings.data?.has_api_key ? 'keep existing (leave blank)' : 'sk-…'}
            autoComplete="off"
          />
          <p className="hint mt-1">
            Stored as-is for v1; wrap with KMS / Vault before shipping to prod
            (tracked in <code className="font-mono">ROADMAP.md</code> V2 § secrets
            encryption).
          </p>
        </div>

        <div>
          <label className="field" htmlFor="baseurl">
            Base URL (optional)
          </label>
          <input
            id="baseurl"
            className="input"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.openai.com/v1   (Azure / Ollama / proxies)"
          />
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            className="h-4 w-4 rounded border-ink-700 bg-ink-900 text-accent-500"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />
          <span className="text-sm text-ink-100">Enabled</span>
        </label>

        {save.error && <p className="error">{errorMessage(save.error)}</p>}

        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            className="btn-secondary"
            disabled={!settings.data?.has_api_key || test.isPending}
            onClick={() => test.mutate()}
          >
            {test.isPending ? 'Testing…' : 'Test connection'}
          </button>
          <button type="submit" className="btn-primary" disabled={!canSave}>
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
        </div>
        {test.error && <p className="error">{errorMessage(test.error)}</p>}
        {test.data && (
          <p className={`text-sm ${test.data.ok ? 'text-success-500' : 'text-danger-500'}`}>
            {test.data.ok
              ? `✓ connected (${test.data.latency_ms?.toFixed(0)}ms)`
              : `✗ ${test.data.error ?? 'failed'}`}
          </p>
        )}
      </form>
    </div>
  );
}
