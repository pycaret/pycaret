/**
 * /workspaces/:wsId/llm — configure the workspace's LLM provider.
 *
 * State-aware UI:
 *  - When a key is on file, the API-key field is hidden behind a
 *    "✓ Key on file" status pill with Rotate + Clear buttons. The form
 *    no longer renders a password input the user has to mentally parse
 *    as "leave blank to keep". This was the misleading bit in v1.
 *  - When no key is on file, an amber "no key on file — LLM features
 *    disabled" status appears above the form, and the API-key input is
 *    required.
 *
 * Backend contract:
 *  - GET  /workspaces/{id}/llm/settings -> includes `has_api_key`.
 *  - PUT  /workspaces/{id}/llm/settings -> PUT-merge; null api_key keeps existing.
 *  - DELETE /workspaces/{id}/llm/settings -> drops the row entirely (used by Clear).
 *  - POST /workspaces/{id}/llm/test-connection -> round-trip probe.
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
  { value: 'anthropic', label: 'Anthropic (Claude)', defaultModel: 'claude-sonnet-4-6', supported: true },
  { value: 'openai', label: 'OpenAI (GPT)', defaultModel: 'gpt-4o-mini', supported: true },
  { value: 'google', label: 'Google Gemini', defaultModel: '', supported: false },
  { value: 'azure_openai', label: 'Azure OpenAI', defaultModel: '', supported: false },
  { value: 'ollama', label: 'Ollama (local)', defaultModel: '', supported: false },
  { value: 'custom_openai_compatible', label: 'Custom OpenAI-compatible', defaultModel: '', supported: false },
];

function fmtRelative(iso?: string | null): string {
  if (!iso) return 'unknown';
  const t = new Date(iso).getTime();
  const ms = Date.now() - t;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

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
  const [modelName, setModelName] = useState('claude-sonnet-4-6');
  const [baseUrl, setBaseUrl] = useState('');
  const [enabled, setEnabled] = useState(true);
  // When a key is stored, the password input is hidden. Rotate flips this on
  // and surfaces the input so the user can paste a replacement.
  const [rotating, setRotating] = useState(false);

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
      setApiKey('');
      setRotating(false);
      qc.invalidateQueries({ queryKey: ['llm', 'settings', wsId] });
    },
  });

  const clear = useMutation({
    mutationFn: () => llmApi.deleteSettings(wsId),
    onSuccess: () => {
      setApiKey('');
      setRotating(false);
      qc.invalidateQueries({ queryKey: ['llm', 'settings', wsId] });
    },
  });

  const test = useMutation({
    mutationFn: () => llmApi.testConnection(wsId),
  });

  const stored = !!settings.data?.has_api_key;
  const providerSupported =
    PROVIDERS.find((p) => p.value === provider)?.supported ?? false;
  const needsKeyInput = !stored || rotating;
  const canSave =
    providerSupported &&
    modelName.trim().length > 0 &&
    (stored || apiKey.trim().length > 0) &&
    !save.isPending;

  return (
    <div className="space-y-6 max-w-2xl">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-900 dark:hover:text-ink-50">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>LLM</span>
        </nav>
        <h1 className="text-xl font-semibold">LLM provider settings</h1>
        <p className="mt-1 text-sm text-ink-500">
          Configure which LLM answers dataset + experiment consultations for this
          workspace. The LLM is <strong>advisory</strong>: it proposes configs +
          risk flags, and the deterministic engine executes what you approve.
        </p>
      </header>

      {/* Status banner — always visible so user knows what state they're in. */}
      {settings.isLoading ? (
        <div className="rounded-md border border-ink-200 dark:border-ink-800 bg-ink-50 dark:bg-ink-950/40 px-3 py-2 text-sm text-ink-500">
          Loading current settings…
        </div>
      ) : stored ? (
        <div className="rounded-md border border-emerald-300/70 dark:border-emerald-700/40 bg-emerald-50 dark:bg-emerald-950/30 px-4 py-3 flex items-start gap-3">
          <CheckBadge />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-emerald-900 dark:text-emerald-200">
              API key on file
            </p>
            <p className="mt-0.5 text-xs text-emerald-800 dark:text-emerald-300/80">
              {settings.data?.provider} · model{' '}
              <code className="font-mono">{settings.data?.model_name}</code> ·
              saved {fmtRelative(settings.data?.created_at)}
              {!settings.data?.enabled && (
                <>
                  {' '}
                  · <span className="font-medium text-amber-700 dark:text-amber-300">disabled</span>
                </>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              className="btn-secondary text-xs"
              disabled={test.isPending}
              onClick={() => test.mutate()}
              title="Round-trip probe the configured provider"
            >
              {test.isPending ? 'Testing…' : 'Test connection'}
            </button>
            <button
              type="button"
              className="btn-secondary text-xs"
              onClick={() => setRotating((v) => !v)}
              title="Replace the stored API key"
            >
              {rotating ? 'Cancel rotation' : 'Rotate key'}
            </button>
            <button
              type="button"
              className="btn-ghost text-xs text-danger-600 hover:text-danger-700"
              disabled={clear.isPending}
              onClick={() => {
                if (
                  window.confirm(
                    'Clear stored LLM settings? The API key + all provider config will be deleted. LLM features will fail until a new key is saved.',
                  )
                ) {
                  clear.mutate();
                }
              }}
              title="Delete the stored key + provider row"
            >
              {clear.isPending ? 'Clearing…' : 'Clear'}
            </button>
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-amber-300 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-950/30 px-4 py-3 flex items-start gap-3">
          <WarnBadge />
          <div className="flex-1">
            <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
              No API key on file
            </p>
            <p className="mt-0.5 text-xs text-amber-800 dark:text-amber-300/80">
              The AI dataset consultant, experiment designer, run debugger, and
              every other LLM-backed feature will fail with 400 until a key is
              saved below.
            </p>
          </div>
        </div>
      )}

      {/* Test result, if any */}
      {test.data && (
        <div
          className={`rounded-md border px-3 py-2 text-sm ${
            test.data.ok
              ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-200'
              : 'border-danger-300 bg-danger-50 dark:bg-danger-950/30 text-danger-700 dark:text-danger-300'
          }`}
        >
          {test.data.ok
            ? `✓ connected in ${test.data.latency_ms?.toFixed(0)}ms (${test.data.provider} · ${test.data.model_name})`
            : `✗ ${test.data.error ?? 'failed'}`}
        </div>
      )}
      {test.error && (
        <p className="error">{errorMessage(test.error)}</p>
      )}
      {clear.error && (
        <p className="error">Clear failed: {errorMessage(clear.error)}</p>
      )}

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
            placeholder={provider === 'anthropic' ? 'claude-sonnet-4-6' : 'gpt-4o-mini'}
            required
          />
        </div>

        {needsKeyInput ? (
          <div>
            <label className="field" htmlFor="apikey">
              {rotating ? 'New API key' : 'API key'}{' '}
              <span className="text-danger-500">*</span>
            </label>
            <input
              id="apikey"
              className="input font-mono"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={provider === 'anthropic' ? 'sk-ant-…' : 'sk-…'}
              autoComplete="off"
              autoFocus={rotating}
              required
            />
            <p className="hint mt-1">
              Stored encrypted at rest via Fernet (PYCARET_SECRETS_KEY).
              KMS/Vault wrapping tracked in <code className="font-mono">ROADMAP.md</code> V2.
            </p>
          </div>
        ) : (
          <div className="rounded-md border border-dashed border-ink-200 dark:border-ink-700 px-3 py-2 text-xs text-ink-500 flex items-center gap-2">
            <LockIcon />
            <span>API key hidden — use <strong>Rotate key</strong> above to replace.</span>
          </div>
        )}

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
            className="h-4 w-4 rounded border-ink-200 bg-white text-accent-500"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />
          <span className="text-sm text-ink-900 dark:text-ink-50">Enabled</span>
        </label>

        {save.error && <p className="error">{errorMessage(save.error)}</p>}

        <div className="flex items-center justify-end gap-2">
          <button type="submit" className="btn-primary" disabled={!canSave}>
            {save.isPending ? 'Saving…' : stored ? 'Save changes' : 'Save'}
          </button>
        </div>
      </form>
    </div>
  );
}

// ─── Icons ───────────────────────────────────────────────────────────

function CheckBadge() {
  return (
    <span className="h-6 w-6 rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 flex items-center justify-center shrink-0 mt-0.5">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M20 6L9 17l-5-5" />
      </svg>
    </span>
  );
}

function WarnBadge() {
  return (
    <span className="h-6 w-6 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 flex items-center justify-center shrink-0 mt-0.5">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M12 9v4 M12 17h.01 M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      </svg>
    </span>
  );
}

function LockIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}
