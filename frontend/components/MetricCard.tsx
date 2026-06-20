"use client";

interface MetricCardProps {
    label: string;
    value: number | string;
    unit?: string;
    icon?: React.ReactNode;
    trend?: "up" | "down" | "neutral";
    size?: "sm" | "md" | "lg";
}

export default function MetricCard({
    label,
    value,
    unit = "",
    icon,
    trend = "neutral",
    size = "md",
}: MetricCardProps) {
    const sizeClasses = {
        sm: "p-3",
        md: "p-4",
        lg: "p-5",
    };

    const valueSizeClasses = {
        sm: "text-lg",
        md: "text-2xl",
        lg: "text-3xl",
    };

    const trendColors = {
        up: "text-emerald-400",
        down: "text-red-400",
        neutral: "text-white",
    };

    const formattedValue = typeof value === "number" ? value.toFixed(4) : value;

    return (
        <div
            className={`
                ${sizeClasses[size]}
                bg-slate-800/50 
                border border-slate-700/50 
                rounded-xl 
                hover:border-slate-600 
                transition-colors
            `}
        >
            <div className="flex items-start justify-between mb-2">
                <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                    {label}
                </span>
                {icon && <div className="text-slate-500">{icon}</div>}
            </div>
            <div className={`font-bold ${valueSizeClasses[size]} ${trendColors[trend]}`}>
                {formattedValue}
                {unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
            </div>
        </div>
    );
}