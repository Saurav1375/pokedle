"use client";

import React, { useState, useEffect } from "react";
import {
  Play,
  RotateCcw,
  Settings,
  ChevronRight,
  ChevronLeft,
  Clock,
  Target,
  Zap,
  Menu,
  X,
} from "lucide-react";

// Types
interface SolverConfig {
  algorithm: string;
  attributes: string[];
  heuristic: string;
  secret_pokemon: string | null;
  max_attempts: number;
}

interface GAConfig {
  pop_size: number;
  elite_size: number;
  mutation_rate: number;
  crossover_rate: number;
  tournament_size: number;
  crossover_strategy: string;
  generations_per_guess: number;
}

interface SolverStep {
  attempt: number;
  guess_name: string;
  guess_data: Record<string, string>;
  feedback: Record<string, string>;
  remaining_candidates: number;
  timestamp: number;
  image_url?: string;
  heuristic_info?: Record<string, any>;
}

interface SolverResult {
  secret_name: string;
  secret_image: string;
  success: boolean;
  total_attempts: number;
  steps: SolverStep[];
  execution_time: number;
}

const API_URL = "http://localhost:8000";

export default function PokedleVisualizer() {
  const [config, setConfig] = useState<SolverConfig>({
    algorithm: "CSP",
    attributes: [
      "Generation",
      "Height",
      "Weight",
      "Type1",
      "Type2",
      "Color",
      "evolutionary_stage",
    ],
    heuristic: "random",
    secret_pokemon: null,
    max_attempts: 10,
  });

  const [gaConfig, setGaConfig] = useState<GAConfig>({
    pop_size: 100,
    elite_size: 20,
    mutation_rate: 0.15,
    crossover_rate: 0.8,
    tournament_size: 7,
    crossover_strategy: "attribute_blend",
    generations_per_guess: 30,
  });

  const [result, setResult] = useState<SolverResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [availableAttrs, setAvailableAttrs] = useState<string[]>([]);
  const [availableHeuristics, setAvailableHeuristics] = useState<
    Record<string, string>
  >({});
  const [availableCrossoverStrategies, setAvailableCrossoverStrategies] =
    useState<Record<string, string>>({});
  const [pokemonList, setPokemonList] = useState<
    Array<{ name: string; image_url: string }>
  >([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);

  // Fetch config options
  useEffect(() => {
    fetch(`${API_URL}/config`)
      .then((res) => res.json())
      .then((data) => {
        setAvailableAttrs(data.attributes);
        setAvailableHeuristics(data.heuristic_descriptions || {});
        setAvailableCrossoverStrategies(data.crossover_strategies || {});
      })
      .catch((err) => console.error("Failed to fetch config:", err));

    fetch(`${API_URL}/pokemon`)
      .then((res) => res.json())
      .then((data) => setPokemonList(data.pokemon || []))
      .catch((err) => console.error("Failed to fetch Pokemon:", err));
  }, []);

  const runSolver = async () => {
    setLoading(true);
    setResult(null);
    setCurrentStep(0);

    try {
      const response = await fetch(`${API_URL}/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...config,
          ga_config: config.algorithm === "GA" ? gaConfig : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error("Solver failed");
      }

      const data: SolverResult = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Error running solver:", err);
      alert("Failed to run solver. Make sure backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const getFeedbackColor = (status: string): string => {
    switch (status) {
      case "green":
        return "bg-green-500";
      case "yellow":
        return "bg-yellow-500";
      case "gray":
        return "bg-gray-400";
      case "higher":
        return "bg-blue-500";
      case "lower":
        return "bg-red-500";
      default:
        return "bg-gray-300";
    }
  };

  const getFeedbackLabel = (status: string): string => {
    switch (status) {
      case "green":
        return "✓";
      case "yellow":
        return "↔";
      case "gray":
        return "✗";
      case "higher":
        return "↑";
      case "lower":
        return "↓";
      default:
        return "?";
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-80" : "w-0"
        } transition-all duration-300 bg-white border-r border-gray-200 flex flex-col overflow-hidden`}
      >
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-gray-700" />
            <h2 className="font-semibold text-gray-900">Settings</h2>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Algorithm Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Algorithm
            </label>
            <div className="grid grid-cols-2 gap-2">
              {["CSP", "GA"].map((algo) => (
                <button
                  key={algo}
                  onClick={() => setConfig({ ...config, algorithm: algo })}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    config.algorithm === algo
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {algo}
                </button>
              ))}
            </div>
          </div>

          {/* Heuristic Selection (CSP only) */}
          {config.algorithm === "CSP" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                CSP Heuristic
              </label>
              <select
                value={config.heuristic}
                onChange={(e) =>
                  setConfig({ ...config, heuristic: e.target.value })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.entries(availableHeuristics).map(([key, desc]) => (
                  <option key={key} value={key}>
                    {key.toUpperCase()} - {desc}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* GA Configuration */}
          {config.algorithm === "GA" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Population: {gaConfig.pop_size}
                </label>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="10"
                  value={gaConfig.pop_size}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      pop_size: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Elite Size: {gaConfig.elite_size}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={gaConfig.elite_size}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      elite_size: parseInt(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Mutation: {(gaConfig.mutation_rate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={gaConfig.mutation_rate}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      mutation_rate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Crossover: {(gaConfig.crossover_rate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={gaConfig.crossover_rate}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      crossover_rate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Crossover Strategy
                </label>
                <select
                  value={gaConfig.crossover_strategy}
                  onChange={(e) =>
                    setGaConfig({
                      ...gaConfig,
                      crossover_strategy: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(availableCrossoverStrategies).map(
                    ([key, desc]) => (
                      <option key={key} value={key}>
                        {key.replace(/_/g, " ")}
                      </option>
                    )
                  )}
                </select>
              </div>
            </div>
          )}

          {/* Attribute Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Attributes ({config.attributes.length})
            </label>
            <div className="space-y-2">
              {availableAttrs.map((attr) => (
                <label
                  key={attr}
                  className="flex items-center gap-2 text-sm cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={config.attributes.includes(attr)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setConfig({
                          ...config,
                          attributes: [...config.attributes, attr],
                        });
                      } else {
                        setConfig({
                          ...config,
                          attributes: config.attributes.filter(
                            (a) => a !== attr
                          ),
                        });
                      }
                    }}
                    className="rounded"
                  />
                  <span className="text-gray-700">{attr}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Secret Pokemon */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Secret Pokemon
            </label>
            <select
              value={config.secret_pokemon || ""}
              onChange={(e) =>
                setConfig({
                  ...config,
                  secret_pokemon: e.target.value || null,
                })
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Random</option>
              {pokemonList.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Max Attempts */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Attempts: {config.max_attempts}
            </label>
            <input
              type="range"
              min="5"
              max="20"
              value={config.max_attempts}
              onChange={(e) =>
                setConfig({
                  ...config,
                  max_attempts: parseInt(e.target.value),
                })
              }
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <Menu className="w-5 h-5 text-gray-600" />
                </button>
              )}
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Pokedle AI Solver
                </h1>
                <p className="text-sm text-gray-500">
                  {config.algorithm} Algorithm Dashboard
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {result && (
                <button
                  onClick={() => {
                    setResult(null);
                    setCurrentStep(0);
                  }}
                  className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
              )}
              <button
                onClick={runSolver}
                disabled={loading || config.attributes.length === 0}
                className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                <Play className="w-4 h-4" />
                {loading ? "Running..." : "Run Solver"}
              </button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {result ? (
            <div className="space-y-6">
              {/* Stats Grid */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Target className="w-5 h-5 text-green-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 font-medium">
                        Result
                      </p>
                      <p className="text-lg font-bold text-gray-900">
                        {result.success ? "Success" : "Failed"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <ChevronRight className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 font-medium">
                        Attempts
                      </p>
                      <p className="text-lg font-bold text-gray-900">
                        {result.total_attempts}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <Clock className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 font-medium">Time</p>
                      <p className="text-lg font-bold text-gray-900">
                        {result.execution_time.toFixed(2)}s
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-orange-100 rounded-lg">
                      <Zap className="w-5 h-5 text-orange-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 font-medium">
                        Algorithm
                      </p>
                      <p className="text-lg font-bold text-gray-900">
                        {config.algorithm}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Secret Pokemon */}
              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200 p-6">
                <div className="flex items-center gap-6">
                  {result.secret_image && (
                    <img
                      src={result.secret_image}
                      alt={result.secret_name}
                      className="w-24 h-24 object-contain bg-white rounded-lg p-2 border border-gray-200"
                    />
                  )}
                  <div>
                    <p className="text-sm text-gray-600 font-medium mb-1">
                      Secret Pokemon
                    </p>
                    <p className="text-3xl font-bold text-gray-900">
                      {result.secret_name}
                    </p>
                  </div>
                </div>
              </div>

              {/* Current Step */}
              <div className="grid grid-cols-3 gap-6">
                {/* Step Details */}
                <div className="col-span-2 bg-white rounded-lg border border-gray-200 p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <p className="text-sm text-gray-500 font-medium">
                        Attempt #{result.steps[currentStep]?.attempt}
                      </p>
                      <h3 className="text-2xl font-bold text-gray-900">
                        {result.steps[currentStep]?.guess_name}
                      </h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() =>
                          setCurrentStep(Math.max(0, currentStep - 1))
                        }
                        disabled={currentStep === 0}
                        className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <span className="text-sm text-gray-600 px-3">
                        {currentStep + 1} / {result.steps.length}
                      </span>
                      <button
                        onClick={() =>
                          setCurrentStep(
                            Math.min(result.steps.length - 1, currentStep + 1)
                          )
                        }
                        disabled={currentStep === result.steps.length - 1}
                        className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {/* Attributes Grid */}
                  <div className="grid grid-cols-3 gap-3">
                    {result.steps[currentStep] &&
                      Object.entries(result.steps[currentStep].guess_data).map(
                        ([attr, value]) => {
                          const feedback =
                            result.steps[currentStep].feedback[attr];
                          return (
                            <div
                              key={attr}
                              className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-500 font-medium">
                                  {attr}
                                </span>
                                <span
                                  className={`w-6 h-6 rounded-full ${getFeedbackColor(
                                    feedback
                                  )} flex items-center justify-center text-white text-xs font-bold`}
                                >
                                  {getFeedbackLabel(feedback)}
                                </span>
                              </div>
                              <p className="text-sm font-semibold text-gray-900">
                                {value}
                              </p>
                            </div>
                          );
                        }
                      )}
                  </div>

                  {/* Heuristic Info */}
                  {result.steps[currentStep]?.heuristic_info && (
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-sm font-medium text-gray-700 mb-2">
                        Algorithm Stats
                      </p>
                      <div className="grid grid-cols-3 gap-3">
                        {Object.entries(
                          result.steps[currentStep].heuristic_info
                        ).map(([key, value]) => (
                          <div key={key}>
                            <span className="text-xs text-gray-500">
                              {key.replace(/_/g, " ")}:
                            </span>
                            <p className="text-sm font-bold text-gray-900">
                              {typeof value === "number"
                                ? value.toFixed(2)
                                : String(value)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Pokemon Image & Timeline */}
                <div className="space-y-4">
                  {result.steps[currentStep]?.image_url && (
                    <div className="bg-white rounded-lg border border-gray-200 p-6 flex items-center justify-center">
                      <img
                        src={result.steps[currentStep].image_url}
                        alt={result.steps[currentStep].guess_name}
                        className="w-48 h-48 object-contain"
                      />
                    </div>
                  )}

                  {/* Timeline */}
                  <div className="bg-white rounded-lg border border-gray-200 p-4 max-h-96 overflow-y-auto">
                    <p className="text-sm font-medium text-gray-700 mb-3">
                      Timeline
                    </p>
                    <div className="space-y-2">
                      {result.steps.map((step, idx) => (
                        <div
                          key={idx}
                          onClick={() => setCurrentStep(idx)}
                          className={`p-3 rounded-lg cursor-pointer transition-colors ${
                            currentStep === idx
                              ? "bg-blue-50 border border-blue-200"
                              : "bg-gray-50 hover:bg-gray-100 border border-gray-200"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-xs text-gray-500">
                                #{step.attempt}
                              </p>
                              <p className="text-sm font-semibold text-gray-900">
                                {step.guess_name}
                              </p>
                            </div>
                            <p className="text-xs text-gray-500">
                              {step.remaining_candidates}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Play className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Ready to solve
                </h3>
                <p className="text-sm text-gray-500">
                  Configure your settings and click "Run Solver" to begin
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
