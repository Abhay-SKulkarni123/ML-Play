"use client";

interface StatsTableProps {
    data: Record<string, any>;
    columns?: string[];
    title?: string;
}

export default function StatsTable({ data, columns, title }: StatsTableProps) {
    if (!data || Object.keys(data).length === 0) return null;

    const entries = Object.entries(data);
    const displayColumns = columns || entries.map(([key]) => key);

    return (
        <div className="w-full">
            {title && (
                <h3 className="text-sm font-semibold text-slate-300 mb-3">{title}</h3>
            )}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-slate-700/50">
                            <th className="text-left py-2 px-3 text-xs font-medium text-slate-400 uppercase tracking-wider">
                                Metric
                            </th>
                            <th className="text-right py-2 px-3 text-xs font-medium text-slate-400 uppercase tracking-wider">
                                Value
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {entries.map(([key, value]) => {
                            if (typeof value === "object" && value !== null) {
                                return (
                                    <tr key={key} className="border-b border-slate-800/50">
                                        <td className="py-2 px-3 text-slate-300 font-medium">
                                            {formatKey(key)}
                                        </td>
                                        <td className="py-2 px-3 text-right">
                                            <StatsTable data={value} />
                                        </td>
                                    </tr>
                                );
                            }

                            const displayValue = formatValue(value);
                            const isPercentage = typeof value === "number" && key.toLowerCase().includes("accuracy");

                            return (
                                <tr key={key} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                                    <td className="py-2.5 px-3 text-slate-300">{formatKey(key)}</td>
                                    <td className={`py-2.5 px-3 text-right font-mono font-medium ${isPercentage ? "text-emerald-400" : "text-white"}`}>
                                        {displayValue}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function formatKey(key: string): string {
    return key
        .replace(/_/g, " ")
        .replace(/\b\w/g, l => l.toUpperCase());
}

function formatValue(value: any): string {
    if (typeof value === "number") {
        if (Number.isInteger(value)) return value.toString();
        return value.toFixed(4);
    }
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (value === null || value === undefined) return "—";
    return String(value);
}