{{/*
Shared Helm template helpers for the PyCaret chart.
Keeps name / label / image / connection-URL logic in one place so the
per-resource templates stay readable.
*/}}

{{- define "pycaret.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "pycaret.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "pycaret.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* api / worker / web names */}}
{{- define "pycaret.api.name" -}}{{ printf "%s-api" (include "pycaret.fullname" .) }}{{- end -}}
{{- define "pycaret.worker.name" -}}{{ printf "%s-worker" (include "pycaret.fullname" .) }}{{- end -}}
{{- define "pycaret.web.name" -}}{{ printf "%s-web" (include "pycaret.fullname" .) }}{{- end -}}
{{- define "pycaret.postgres.name" -}}{{ printf "%s-postgres" (include "pycaret.fullname" .) }}{{- end -}}
{{- define "pycaret.redis.name" -}}{{ printf "%s-redis" (include "pycaret.fullname" .) }}{{- end -}}
{{- define "pycaret.minio.name" -}}{{ printf "%s-minio" (include "pycaret.fullname" .) }}{{- end -}}

{{- define "pycaret.image" -}}
{{- $tag := default .Values.global.imageTag .image.tag -}}
{{ .image.repository }}:{{ $tag }}
{{- end -}}

{{/* Connection URLs. Resolved to the in-cluster service when the
    relevant sub-chart is enabled, otherwise the externalX values. */}}

{{- define "pycaret.databaseUrl" -}}
{{- if .Values.postgres.enabled -}}
postgresql+psycopg://{{ .Values.postgres.username }}:$(POSTGRES_PASSWORD)@{{ include "pycaret.postgres.name" . }}:5432/{{ .Values.postgres.database }}
{{- else -}}
{{ .Values.externalPostgres.url }}
{{- end -}}
{{- end -}}

{{- define "pycaret.redisUrl" -}}
{{- if .Values.redis.enabled -}}
redis://{{ include "pycaret.redis.name" . }}:6379/0
{{- else -}}
{{ .Values.externalRedis.url }}
{{- end -}}
{{- end -}}

{{- define "pycaret.s3Endpoint" -}}
{{- if .Values.minio.enabled -}}
http://{{ include "pycaret.minio.name" . }}:9000
{{- else -}}
{{ .Values.externalS3.endpoint }}
{{- end -}}
{{- end -}}

{{- define "pycaret.s3Bucket" -}}
{{- if .Values.minio.enabled -}}
{{ .Values.minio.bucket }}
{{- else -}}
{{ .Values.externalS3.bucket }}
{{- end -}}
{{- end -}}
