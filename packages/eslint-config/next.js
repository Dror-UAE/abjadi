import { config as react } from "./react.js";

/** @type {import("eslint").Linter.Config[]} */
export const config = [
  ...react,
  {
    rules: {
      "react/react-in-jsx-scope": "off",
    },
  },
];
