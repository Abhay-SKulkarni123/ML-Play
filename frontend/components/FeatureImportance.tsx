"use client";

interface FeatureImportanceProps {
    data: Record<string, number> | null;
    title?: string;
    maxFeatures?: number;
}

export default function FeatureImportance({ data, title = "Feature Importance", maxFeatures = 10 }: FeatureImportanceProps) {
    if (!data || Object.keys(data).length === 0) return null;

    // Sort by importance and take top N
    const sorted = Object.entries(data)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, maxFeatures);

    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)));

    return (
        <div className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">{title}</h3>
            <div className="space-y-2.5">
                {sorted.map(([feature, importance]) => {
                    const width = (Math.abs(importance) / maxAbs) * 100;
                    const isPositive = importance >= 0;

                    return (
                        <div key={feature} className="flex items-center gap-3">
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs text-slate-300 truncate" title={feature}>
                                        {formatFeatureName(feature)}
                                    </span>
                                    <span className="text-xs font-mono text-slate-400 ml-2">
                                        {importance.toFixed(4)}
                                    </span>
                                </div>
                                <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all ${isPositive ? "bg-blue-500" : "bg-amber-500"}`}
                                        style={{ width: `${width}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
            <div className="mt-4 flex items-center gap-4 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-blue-500" />
                    <span>Positive impact</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-amber-500" />
                    <span>Negative impact</span>
                </div>
            </div>
        </div>
    );
}

function formatFeatureName(name: string): string {
    return name
        .replace(/_/g, " ")
        .replace(/\b\w/g, l => l.toUpperCase());
}