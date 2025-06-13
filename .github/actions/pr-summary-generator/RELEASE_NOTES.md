# Release Notes

## v1.0.1 - Initial Rerelease

### Features

-   Automatic PR summary generation using Ollama LLM
-   Triggers on PR open and update events
-   Easy integration with any repository

### Technical Details

-   Uses Python 3.13
-   Integrates with Ollama for LLM processing
-   Includes comprehensive documentation

### Requirements

-   GitHub repository with Actions enabled
-   Ollama service URL (set as repository secret)
-   GitHub token (automatically provided)

### Breaking Changes

None - This is the initial release.

### Known Issues

None at this time.

### Security

-   Uses GitHub's built-in token system
-   Requires explicit permission settings
-   No sensitive data is stored or logged
