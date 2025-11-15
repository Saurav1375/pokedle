import React, { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Zap,
  Users,
  GitMerge,
  Shuffle,
  Trophy,
  TrendingUp,
  Activity,
} from "lucide-react";

// Types
interface PokemonAttributes {
  [key: string]: string | number;
}

interface Pokemon {
  name: string;
  fitness: number;
  attributes: PokemonAttributes;
  index: number;
  image_url?: string;
}

interface SelectionPair {
  parent1: Pokemon;
  parent2: Pokemon;
}

interface CrossoverResult {
  offspring: Pokemon;
  is_new: boolean;
}

interface MutationResult {
  before: Pokemon;
  after: Pokemon;
  mutated: boolean;
}

interface FitnessStats {
  max: number;
  avg: number;
  median: number;
  min: number;
}

interface GenerationData {
  generation_number: number;
  fitness_stats: FitnessStats;
  elite_preserved: Pokemon[];
  new_population: Pokemon[];
  selection_pairs: SelectionPair[];
  crossover_results: CrossoverResult[];
  mutation_results: MutationResult[];
}

interface GAVisualizationProps {
  generationHistory: GenerationData[];
}

type TabId = "overview" | "selection" | "crossover" | "mutation";

const GAVisualization: React.FC<GAVisualizationProps> = ({
  generationHistory,
}) => {
  const [expandedGen, setExpandedGen] = useState<number | null>(null);
  const [selectedTab, setSelectedTab] = useState<TabId>("overview");

  if (!generationHistory || generationHistory.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No generation data available</p>
      </div>
    );
  }

  const toggleGeneration = (genNum: number) => {
    setExpandedGen(expandedGen === genNum ? null : genNum);
  };

  interface PokemonCardProps {
    pokemon: Pokemon;
    label?: string;
    highlight?: boolean;
  }

  const PokemonCard: React.FC<PokemonCardProps> = ({
    pokemon,
    label,
    highlight = false,
  }) => (
    <div
      className={`relative p-3 rounded border transition-all ${
        highlight
          ? "border-2 border-black bg-gray-50"
          : "border border-gray-300 bg-white hover:border-gray-400"
      }`}
    >
      {label && (
        <div className="absolute -top-2 left-2 px-2 py-0.5 bg-black text-white text-xs font-semibold rounded">
          {label}
        </div>
      )}
      <div className="flex items-center gap-2 mb-2">
        {pokemon.image_url && (
          <img
            src={pokemon.image_url}
            alt={pokemon.name}
            className="w-12 h-12 object-contain bg-gray-50 rounded border border-gray-200"
          />
        )}
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm text-black truncate">
            {pokemon.name}
          </p>
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-black" />
            <span className="text-xs font-medium text-gray-700">
              {pokemon.fitness}
            </span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-1">
        {Object.entries(pokemon.attributes)
          .slice(0, 4)
          .map(([key, val]) => (
            <div key={key} className="text-xs">
              <span className="text-gray-500">{key}:</span>
              <span className="font-medium text-black ml-1">{val}</span>
            </div>
          ))}
      </div>
    </div>
  );

  interface FitnessBarProps {
    value: number;
    max?: number;
  }

  const FitnessBar: React.FC<FitnessBarProps> = ({ value, max = 100 }) => {
    const percentage = (value / max) * 100;

    return (
      <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
        <div
          className="h-2 rounded-full transition-all duration-500 bg-black"
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    );
  };

  interface GenerationCardProps {
    gen: GenerationData;
    index: number;
  }

  const GenerationCard: React.FC<GenerationCardProps> = ({ gen, index }) => {
    const isExpanded = expandedGen === gen.generation_number;

    return (
      <div className="bg-white rounded border-2 border-gray-300 overflow-hidden hover:border-gray-400 transition-all">
        {/* Header */}
        <button
          onClick={() => toggleGeneration(gen.generation_number)}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-black rounded-full flex items-center justify-center text-white font-bold text-lg">
              {gen.generation_number}
            </div>
            <div className="text-left">
              <p className="font-bold text-black">
                Generation {gen.generation_number}
              </p>
              <div className="flex items-center gap-4 mt-1">
                <div className="flex items-center gap-1">
                  <Trophy className="w-3 h-3 text-black" />
                  <span className="text-xs text-gray-600">
                    Max: {gen.fitness_stats.max}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 text-black" />
                  <span className="text-xs text-gray-600">
                    Avg: {gen.fitness_stats.avg}
                  </span>
                </div>
              </div>
            </div>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>

        {/* Expanded Content */}
        {isExpanded && (
          <div className="border-t border-gray-300 p-6 bg-gray-50">
            {/* Tabs */}
            <div className="flex gap-2 mb-6 border-b border-gray-300">
              {[
                { id: "overview" as TabId, label: "Overview", icon: Activity },
                { id: "selection" as TabId, label: "Selection", icon: Users },
                {
                  id: "crossover" as TabId,
                  label: "Crossover",
                  icon: GitMerge,
                },
                { id: "mutation" as TabId, label: "Mutation", icon: Shuffle },
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setSelectedTab(id)}
                  className={`flex items-center gap-2 px-4 py-2 font-medium text-sm transition-colors ${
                    selectedTab === id
                      ? "text-black border-b-2 border-black"
                      : "text-gray-500 hover:text-black"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </div>

            {/* Overview Tab */}
            {selectedTab === "overview" && (
              <div className="space-y-6">
                {/* Fitness Stats */}
                <div className="grid grid-cols-4 gap-4">
                  {[
                    {
                      label: "Max",
                      value: gen.fitness_stats.max,
                    },
                    {
                      label: "Avg",
                      value: gen.fitness_stats.avg,
                    },
                    {
                      label: "Median",
                      value: gen.fitness_stats.median,
                    },
                    {
                      label: "Min",
                      value: gen.fitness_stats.min,
                    },
                  ].map(({ label, value }) => (
                    <div
                      key={label}
                      className="bg-white rounded border border-gray-300 p-4"
                    >
                      <p className="text-xs text-gray-500 mb-1">
                        {label} Fitness
                      </p>
                      <p className="text-2xl font-bold text-black">{value}</p>
                      <FitnessBar value={value} />
                    </div>
                  ))}
                </div>

                {/* Elite Preserved */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Trophy className="w-5 h-5 text-black" />
                    <h4 className="font-bold text-black">
                      Elite Preserved (Top 5)
                    </h4>
                  </div>
                  <div className="grid grid-cols-5 gap-3">
                    {gen.elite_preserved.map((pokemon, idx) => (
                      <PokemonCard
                        key={pokemon.index}
                        pokemon={pokemon}
                        label={idx === 0 ? "👑 Best" : `#${idx + 1}`}
                        highlight={idx === 0}
                      />
                    ))}
                  </div>
                </div>

                {/* Population Sample */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Users className="w-5 h-5 text-black" />
                    <h4 className="font-bold text-black">
                      New Population Sample (Top 10)
                    </h4>
                  </div>
                  <div className="grid grid-cols-5 gap-3">
                    {gen.new_population.map((pokemon) => (
                      <PokemonCard key={pokemon.index} pokemon={pokemon} />
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Selection Tab */}
            {selectedTab === "selection" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <Users className="w-5 h-5 text-black" />
                  <h4 className="font-bold text-black">
                    Tournament Selection (First 5 Pairs)
                  </h4>
                </div>
                {gen.selection_pairs.map((pair, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded border border-gray-300 p-4"
                  >
                    <div className="flex items-center gap-1 mb-3">
                      <span className="px-2 py-1 bg-gray-200 text-black text-xs font-semibold rounded">
                        Pair {idx + 1}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <PokemonCard pokemon={pair.parent1} label="Parent 1" />
                      <PokemonCard pokemon={pair.parent2} label="Parent 2" />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Crossover Tab */}
            {selectedTab === "crossover" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <GitMerge className="w-5 h-5 text-black" />
                  <h4 className="font-bold text-black">
                    Crossover Results (First 5)
                  </h4>
                </div>
                {gen.crossover_results.map((result, idx) => {
                  const pair = gen.selection_pairs[idx];
                  return (
                    <div
                      key={idx}
                      className="bg-white rounded border-2 border-gray-400 p-4"
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <span className="px-2 py-1 bg-gray-200 text-black text-xs font-semibold rounded">
                          Crossover {idx + 1}
                        </span>
                        {result.is_new && (
                          <span className="px-2 py-1 bg-black text-white text-xs font-semibold rounded">
                            ✨ New Pokemon
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-3 gap-4 items-center">
                        <PokemonCard pokemon={pair.parent1} label="P1" />
                        <div className="flex flex-col items-center justify-center">
                          <GitMerge className="w-8 h-8 text-black mb-2" />
                          <span className="text-xs text-gray-600 font-medium">
                            Crossover
                          </span>
                        </div>
                        <PokemonCard
                          pokemon={result.offspring}
                          label="Offspring"
                          highlight={result.is_new}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Mutation Tab */}
            {selectedTab === "mutation" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <Shuffle className="w-5 h-5 text-black" />
                  <h4 className="font-bold text-black">
                    Mutation Results (First 5)
                  </h4>
                </div>
                {gen.mutation_results.map((result, idx) => (
                  <div
                    key={idx}
                    className={`rounded border-2 p-4 ${
                      result.mutated
                        ? "bg-gray-100 border-black"
                        : "bg-white border-gray-300"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded ${
                          result.mutated
                            ? "bg-black text-white"
                            : "bg-gray-200 text-gray-600"
                        }`}
                      >
                        {result.mutated ? "🧬 Mutated" : "Unchanged"}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-4 items-center">
                      <PokemonCard pokemon={result.before} label="Before" />
                      <div className="flex flex-col items-center justify-center">
                        <Shuffle
                          className={`w-8 h-8 mb-2 ${
                            result.mutated ? "text-black" : "text-gray-400"
                          }`}
                        />
                        <span className="text-xs text-gray-600 font-medium">
                          Mutation
                        </span>
                      </div>
                      <PokemonCard
                        pokemon={result.after}
                        label="After"
                        highlight={result.mutated}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4 max-w-7xl mx-auto p-4">
      <div className="bg-black rounded border-2 border-black p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-bold mb-2">
              Genetic Algorithm Evolution
            </h3>
            <p className="text-gray-300">
              {generationHistory.length} generations evolved • Best Fitness:{" "}
              {Math.max(...generationHistory.map((g) => g.fitness_stats.max))}
            </p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold">{generationHistory.length}</div>
            <div className="text-sm text-gray-300">Generations</div>
          </div>
        </div>
      </div>

      {/* Fitness Progress Chart */}
      <div className="bg-white rounded border-2 border-gray-300 p-6">
        <h4 className="font-bold text-black mb-4">Fitness Evolution</h4>
        <div className="relative h-32">
          <svg
            className="w-full h-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {/* Max fitness line */}
            <polyline
              fill="none"
              stroke="#000000"
              strokeWidth="2"
              points={generationHistory
                .map(
                  (g, i) =>
                    `${(i / (generationHistory.length - 1)) * 100},${
                      100 - g.fitness_stats.max
                    }`
                )
                .join(" ")}
            />
            {/* Avg fitness line */}
            <polyline
              fill="none"
              stroke="#6b7280"
              strokeWidth="2"
              strokeDasharray="4"
              points={generationHistory
                .map(
                  (g, i) =>
                    `${(i / (generationHistory.length - 1)) * 100},${
                      100 - g.fitness_stats.avg
                    }`
                )
                .join(" ")}
            />
          </svg>
          <div className="absolute top-0 right-0 flex gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-black"></div>
              <span className="text-gray-600">Max</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-gray-500 border-t border-dashed border-gray-500"></div>
              <span className="text-gray-600">Avg</span>
            </div>
          </div>
        </div>
      </div>

      {/* Generation Cards */}
      <div className="space-y-3">
        {generationHistory.map((gen, idx) => (
          <GenerationCard key={gen.generation_number} gen={gen} index={idx} />
        ))}
      </div>
    </div>
  );
};

export default GAVisualization;
