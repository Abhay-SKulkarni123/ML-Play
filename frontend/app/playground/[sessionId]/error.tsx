"use client";
import { useEffect } from "react";
import { AlertTriangle } from "lucide-react";

export default function PlaygroundError({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    useEffect(() => {
        console.error("Playground error:", error);
    }, [error]);

    return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center px-6">
            <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center">
                <div className="flex justify-center mb-4">
                    <div className="p-3 rounded-full bg-red-500/10 border border-red-500/20">
                        <AlertTriangle className="w-6 h-6 text-red-400" />
                    </div>
                </div>
                <h2 className="text-white font-semibold text-lg mb-2">
                    Something went wrong
                </h2>
                <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                    {error.message || "An unexpected error occurred in the playground."}
                </p>
                <div className="flex gap-3 justify-center">
                    <button
                        onClick={reset}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-500
              text-white text-sm font-medium rounded-lg transition-colors">
                        Try again
                    </button>
                    <button
                        onClick={() => window.location.href = "/"}
                        className="px-4 py-2 bg-slate-800 hover:bg-slate-700
              text-slate-300 text-sm font-medium rounded-lg
              transition-colors border border-slate-700">
                        Back to home
                    </button>
                </div>
            </div>
        </div>
    );
}