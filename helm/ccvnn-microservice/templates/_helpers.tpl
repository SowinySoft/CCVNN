{{/*
Expand the name of the chart.
*/}}
{{- define "ccvnn-microservice.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ccvnn-microservice.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "ccvnn-microservice.labels" -}}
helm.sh/chart: {{ include "ccvnn-microservice.chart" . }}
{{ include "ccvnn-microservice.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "ccvnn-microservice.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ccvnn-microservice.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
