// This file is used to lint our commit messages with commitlint
// https://commitlint.js.org/

// Each commit message consists of a header, a body and a footer.
//   <header>
//   <BLANK LINE>
//   <body>
//   <BLANK LINE>
//   <footer>
//

module.exports = {
  rules: {
    "header-case": [2, "always", "sentence-case"],
    "header-max-length": [2, "always", 100],
    "header-full-stop": [2, "never", "."],
  },
};
