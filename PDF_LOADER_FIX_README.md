# PDF Loader Dependency Compatibility Fix

## Issue Description

This fix addresses dependency compatibility issues with PDF loaders in LangChain Community, specifically:

1. **Issue #32360**: `ModuleNotFoundError: No module named 'pdfminer.layout'`
2. Import errors with `pdfminer.utils.open_filename` 
3. Confusion between `pdfminer` and `pdfminer.six` packages

## Root Cause

The issues occurred due to:

1. **Package confusion**: Users installing `pdfminer` instead of `pdfminer.six`
2. **Version incompatibilities**: Older versions of `pdfminer.six` and `unstructured` having import conflicts
3. **Import location changes**: Functions like `open_filename` being moved between modules in different pdfminer versions

## Solution Implemented

### 1. Enhanced Error Messages in `UnstructuredPDFLoader`

```python
def _get_elements(self) -> list:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as e:
        # Provide more helpful error messages for common dependency issues
        error_msg = str(e)
        if "pdfminer.layout" in error_msg:
            raise ImportError(
                "Failed to import unstructured due to pdfminer dependency issues. "
                "Please ensure you have the correct versions installed:\n"
                "1. Install/upgrade pdfminer.six: pip install --upgrade 'pdfminer.six>=20221105'\n"
                "2. Install/upgrade unstructured with PDF support: pip install --upgrade 'unstructured[pdf]>=0.15'\n"
                "Note: Make sure you install 'pdfminer.six' (not 'pdfminer')."
            ) from e
        # ... additional error cases
```

### 2. Import Fallback Logic in `PDFMinerPDFasHTMLLoader`

```python
def lazy_load(self) -> Iterator[Document]:
    # Try to import open_filename from multiple locations for compatibility
    try:
        from pdfminer.utils import open_filename
    except ImportError:
        try:
            from pdfminer.high_level import open_filename
        except ImportError:
            raise ImportError(
                "Could not import 'open_filename' from pdfminer. "
                "Please make sure you have installed the correct version of pdfminer.six. "
                "Try: pip install --upgrade pdfminer.six"
            )
```

## Benefits

1. **Clear Error Messages**: Users get actionable error messages instead of cryptic import errors
2. **Correct Package Names**: Errors explicitly mention `pdfminer.six` vs `pdfminer`
3. **Installation Instructions**: Step-by-step commands to fix dependency issues
4. **Version Compatibility**: Import fallback handles different pdfminer.six versions
5. **Backward Compatibility**: Existing working code continues to work

## User Impact

### Before the Fix
```
ModuleNotFoundError: No module named 'pdfminer.layout'
ImportError: cannot import name 'open_filename' from 'pdfminer.utils'
```

### After the Fix
```
ImportError: Failed to import unstructured due to pdfminer dependency issues. 
Please ensure you have the correct versions installed:
1. Install/upgrade pdfminer.six: pip install --upgrade 'pdfminer.six>=20221105'
2. Install/upgrade unstructured with PDF support: pip install --upgrade 'unstructured[pdf]>=0.15'
Note: Make sure you install 'pdfminer.six' (not 'pdfminer').
```

## Recommended Dependencies

For users experiencing PDF loading issues:

```bash
# Uninstall any conflicting packages
pip uninstall pdfminer pdfminer.six

# Install the correct versions
pip install 'pdfminer.six>=20221105'
pip install 'unstructured[pdf]>=0.15'
```

## Testing

The fix has been tested with:
- Import error simulation
- Fallback logic verification
- Error message validation
- Backward compatibility

## Files Modified

- `libs/community/langchain_community/document_loaders/pdf.py`
  - Enhanced `UnstructuredPDFLoader._get_elements()` error handling
  - Added `PDFMinerPDFasHTMLLoader.lazy_load()` import fallback

## Future Considerations

1. Monitor for new pdfminer.six releases that might change import locations
2. Consider pinning compatible versions in requirements files
3. Add integration tests for different dependency versions
4. Document dependency requirements more clearly in setup guides

## Related Issues

- Fixes #32360: ModuleNotFoundError: No module named 'pdfminer.layout'
- Addresses broader compatibility issues between unstructured and pdfminer packages
- Improves user experience with PDF document loading in LangChain
