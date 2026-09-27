import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "dist",
      "mock-public",
      "public",
      "test-results",
      "playwright-report",
      "src/api/generated.ts",
    ],
  },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2023,
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Only src/api/client.ts may talk to the network.
      "no-restricted-globals": [
        "error",
        { name: "fetch", message: "Call the API through src/api/client.ts." },
      ],
    },
  },
  {
    files: ["src/api/client.ts", "src/mocks/**", "e2e/**", "src/**/*.test.{ts,tsx}"],
    rules: { "no-restricted-globals": "off" },
  },
);
