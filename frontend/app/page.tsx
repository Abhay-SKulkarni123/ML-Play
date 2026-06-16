"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getDatasets, createSession, Dataset } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Brain, Database, Loader2, Rows3, Columns3 } from "lucide-react";

export default function Home() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    getDatasets().then(setDatasets).finally(() => setLoading(false));
  }, []);

  async function start(datasetId: string) {
    setStarting(datasetId);
    try {
      const session = await createSession(datasetId);
      router.push(`/playground/${session.id}`);
    } catch (e) {
      console.error(e);
      setStarting(null);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
      <div className="max-w-5xl mx-auto px-6 py-16">

        {/* Header */}
        <div className="text-center mb-14">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-2xl bg-blue-500/10 border border-blue-500/20">
              <Brain className="w-8 h-8 text-blue-400" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-3 tracking-tight">
            ML Playground
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Walk through the complete machine learning lifecycle step by step.
            Every decision explained by AI.
          </p>
        </div>

        {/* Dataset grid */}
        {loading ? (
          <div className="flex justify-center mt-20">
            <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {datasets.map((ds) => (
              <Card key={ds.id}
                className="bg-slate-800/60 border-slate-700/60 hover:border-blue-500/50 hover:bg-slate-800 transition-all duration-200 group">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="p-2 rounded-lg bg-slate-700/50 group-hover:bg-blue-500/10 transition-colors">
                      <Database className="w-4 h-4 text-slate-400 group-hover:text-blue-400 transition-colors" />
                    </div>
                    <Badge variant="secondary"
                      className={ds.task === "classification"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : "bg-amber-500/10 text-amber-400 border-amber-500/20"}>
                      {ds.task}
                    </Badge>
                  </div>
                  <CardTitle className="text-white text-lg mt-3">{ds.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-4 text-sm text-slate-400 mb-5">
                    <span className="flex items-center gap-1.5">
                      <Rows3 className="w-3.5 h-3.5" />{ds.rows.toLocaleString()} rows
                    </span>
                    <span className="flex items-center gap-1.5">
                      <Columns3 className="w-3.5 h-3.5" />{ds.cols} cols
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mb-4">
                    Target: <span className="text-slate-300 font-mono">{ds.target}</span>
                  </div>
                  <Button className="w-full bg-blue-600 hover:bg-blue-500 text-white"
                    onClick={() => start(ds.id)}
                    disabled={starting === ds.id}>
                    {starting === ds.id
                      ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />Starting...</>
                      : "Start Playground →"}
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}