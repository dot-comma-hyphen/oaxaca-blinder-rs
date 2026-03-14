# Contributing Guidelines

Thank you for your interest in contributing to `oaxaca-blinder-rs`! As a project focused on statistical rigor and numerical stability, we maintain high standards for code and documentation.

## 🤝 How to Contribute

1.  **Search Issues**: Check for existing issues or discussions before starting new work.
2.  **Fork & Branch**: Fork the repo and create a branch named `feature/your-feature` or `fix/your-fix`.
3.  **Implement**: Write your code following the [Development Guide](DEVELOPMENT.md).
4.  **Test**: Ensure all tests pass (`cargo test`). Add new tests for new features.
5.  **Submit PR**: Describe your changes clearly and link to any relevant issues.

## 📐 Code Review Standards

When reviewing Pull Requests, we look for:
- **Numerical Correctness**: Do the math changes align with peer-reviewed econometric literature?
- **Stability**: Does the code handle edge cases like singular matrices or empty groups?
- **Safety**: Is the code free of `unsafe` blocks?
- **Readability**: Are complex algorithms well-commented?
- **Tests**: Is there sufficient unit and integration coverage?

## 📚 Documentation Updates

If your change affects the public API or tool definitions:
- Update [API.md](API.md).
- Update the crate's doc-comments.
- Add an entry to [CHANGELOG.md](../CHANGELOG.md).

## ⚖️ License
By contributing, you agree that your contributions will be licensed under the project's **MIT License**.
