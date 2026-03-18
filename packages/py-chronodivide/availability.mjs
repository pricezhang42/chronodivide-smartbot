function safeNumber(value, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function toSortedValues(mapLike) {
  if (!(mapLike instanceof Map)) {
    return [];
  }
  return Array.from(mapLike.values()).sort((left, right) => String(left?.name ?? "").localeCompare(String(right?.name ?? "")));
}

function buildTechTreeNode(rules, objectGroup) {
  if (!rules?.name) {
    return null;
  }

  return {
    name: rules.name,
    objectGroup,
    type: rules.type,
    techLevel: safeNumber(rules.techLevel, -1),
    buildCat: rules.buildCat ?? null,
    buildLimit: safeNumber(rules.buildLimit, 0),
    prerequisite: Array.isArray(rules.prerequisite) ? rules.prerequisite.slice() : [],
    prerequisiteOverride: Array.isArray(rules.prerequisiteOverride) ? rules.prerequisiteOverride.slice() : [],
    owner: Array.isArray(rules.owner) ? rules.owner.slice() : [],
    requiresStolenAlliedTech: Boolean(rules.requiresStolenAlliedTech),
    requiresStolenSovietTech: Boolean(rules.requiresStolenSovietTech),
    requiresStolenThirdTech: Boolean(rules.requiresStolenThirdTech),
    cost: safeNumber(rules.cost),
    power: safeNumber(rules.power),
    powered: Boolean(rules.powered),
    adjacent: safeNumber(rules.adjacent),
    constructionYard: Boolean(rules.constructionYard),
    refinery: Boolean(rules.refinery),
    factory: rules.factory ?? null,
    weaponsFactory: Boolean(rules.weaponsFactory),
    helipad: Boolean(rules.helipad),
    naval: Boolean(rules.naval),
    superWeapon: rules.superWeapon ?? null,
    availableToMultiplayer: Array.isArray(rules.owner) && rules.owner.length > 0 && safeNumber(rules.techLevel, -1) >= 0,
  };
}

function buildGroupEntries(rulesMap, objectGroup) {
  return toSortedValues(rulesMap)
    .map((rules) => buildTechTreeNode(rules, objectGroup))
    .filter(Boolean);
}

export function buildStaticTechTree(gameApi) {
  const rules = gameApi?.rules;
  if (!rules) {
    return {
      version: "v1",
      notes: ["RulesApi was unavailable; static tech tree could not be built."],
      prereqCategories: {},
      multiplayerCountries: [],
      entries: [],
    };
  }

  const general = rules.general;
  const prereqCategories = {};
  if (general?.prereqCategories instanceof Map) {
    for (const [categoryId, names] of general.prereqCategories.entries()) {
      prereqCategories[String(categoryId)] = Array.isArray(names) ? names.slice() : [];
    }
  }

  const multiplayerCountries = Array.isArray(rules.getMultiplayerCountries?.())
    ? rules.getMultiplayerCountries().map((country) => ({
        name: country?.name,
        side: country?.side ?? null,
        multyplay: country?.multiplay ?? null,
      }))
    : [];

  const entries = [
    ...buildGroupEntries(rules.buildingRules, "building"),
    ...buildGroupEntries(rules.infantryRules, "infantry"),
    ...buildGroupEntries(rules.vehicleRules, "vehicle"),
    ...buildGroupEntries(rules.aircraftRules, "aircraft"),
  ].sort((left, right) => left.name.localeCompare(right.name));

  return {
    version: "v1",
    notes: [
      "Static tech-tree entries are extracted from RulesApi and are not player-state specific.",
      "Dynamic action availability still depends on current player state, queues, placement legality, and selection.",
    ],
    prereqCategories,
    multiplayerCountries,
    entryCount: entries.length,
    entries,
  };
}
