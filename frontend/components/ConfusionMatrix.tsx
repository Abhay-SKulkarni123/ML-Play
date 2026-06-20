"use client";

interface ConfusionMatrixProps {
    matrix: number[][] | null;
    labels?: string[];
    size?: "sm" | "md" | "lg";
}

export default function ConfusionMatrix({ matrix, labels, size = "md" }: ConfusionMatrixProps) {
    if (!matrix || matrix.length === 0) return null;

    const sizeClasses = {
        sm: "p-3",
        md: "p-5",
        lg: "p-6",
    };

    const cellSizeClasses = {
        sm: "w-12 h-12 text-xs",
        md: "w-16 h-16 text-sm",
        lg: "w-20 h-20 text-base",
    };

    const maxVal = Math.max(...matrix.flat());
    const n = matrix.length;

    return (
        <div className={`${sizeClasses[size]} bg-slate-800/50 border border-slate-700/50 rounded-xl`}>
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Confusion Matrix</h3>
            <div className="inline-block">
                <div className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${n}, 1fr)` }}>
                    {/* Header row */}
                    <div className="w-10" /> {/* Empty corner */}
                    {labels?.map((label, i) => (
                        <div key={i} className={`${cellSizeClasses[size]} flex items-center justify-center text-xs font-medium text-slate-400`}>
                            {label}
                        </div>
                    ))}

                    {/* Matrix rows */}
                    {matrix.map((row, i) => (
                        <>
                            <div key={`label-${i}`} className={`${cellSizeClasses[size]} flex items-center justify-center text-xs font-medium text-slate-400`}>
                                {labels?.[i] || `Class ${i}`}
                            </div>
                            {row.map((value, j) => {
                                const intensity = value / maxVal;
                                const isCorrect = i === j;
                                const bgColor = isCorrect
                                    ? `rgba(16, 185, 129, ${0.2 + intensity * 0.6})`
                                    : `rgba(239, 68, 68, ${0.2 + intensity * 0.6})`;

                                return (
                                    <div
                                        key={j}
                                        className={`
                                            ${cellSizeClasses[size]}
                                            flex flex-col items-center justify-center
                                            rounded-lg border border-slate-700/30
                                            font-mono font-bold
                                        `}
                                        style={{ backgroundColor: bgColor }}
                                    >
                                        <span className={isCorrect ? "text-emerald-300" : "text-red-300"}>
                                            {value}
                                        </span>
                                        {isCorrect && (
                                            <span className="text-[10px] text-emerald-400/80 mt-0.5">
                                                Correct
                                            </span>
                                        )}
                                        {!isCorrect && (
                                            <span className="text-[10px] text-red-400/80 mt-0.5">
                                                Error
                                            </span>
                                        )}
                                    </div>
                                );
                            })}
                        </>
                    ))}
                </div>
            </div>
            <div className="mt-3 flex items-center gap-4 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-emerald-500/40" />
                    <span>Correct</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-red-500/40" />
                    <span>Error</span>
                </div>
            </div>
        </div>
    );
}