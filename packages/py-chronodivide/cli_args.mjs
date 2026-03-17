function coerceScalar(value) {
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  return value;
}

export function parseArgs(argv) {
  const args = { _: [] };

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      args._.push(token);
      continue;
    }

    const body = token.slice(2);
    const eqIndex = body.indexOf("=");
    if (eqIndex !== -1) {
      const key = body.slice(0, eqIndex);
      const value = body.slice(eqIndex + 1);
      args[key] = coerceScalar(value);
      continue;
    }

    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      args[body] = coerceScalar(next);
      i += 1;
    } else {
      args[body] = true;
    }
  }

  return args;
}

export function parseIntegerArg(value, fallback, name) {
  if (value === undefined) {
    return fallback;
  }

  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`Expected --${name} to be an integer, got "${value}".`);
  }

  return parsed;
}

export function parseCsvIntegerArg(value) {
  if (value === undefined || value === "") {
    return [];
  }

  return String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      const parsed = Number(item);
      if (!Number.isInteger(parsed)) {
        throw new Error(`Expected a comma-separated integer list, got "${value}".`);
      }
      return parsed;
    });
}

export function requireArg(args, key) {
  const value = args[key];
  if (value === undefined || value === true) {
    throw new Error(`Missing required argument --${key}.`);
  }
  return String(value);
}
