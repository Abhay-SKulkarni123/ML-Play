export default function PlaygroundLoading() {
    return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
                <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent
          rounded-full animate-spin" />
                <p className="text-slate-500 text-sm">Loading playground...</p>
            </div>
        </div>
    );
}