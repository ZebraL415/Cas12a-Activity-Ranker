# After PR approval

The review branch and archives are ready. PR creation is deferred at the
maintainer's request. After signing in, open
https://github.com/ZebraL415/Cas12a-Activity-Ranker/compare/main...codex/paper-reproducibility?expand=1
to create the PR. The review bundle supplies a ready-to-use `PR_DRAFT.md`.
These are the subsequent steps after approving its code, archives and manuscript replacements.

1. Merge the PR and record the actual merged commit SHA. Keep the original
   default model unchanged.
2. Repackage only metadata and manifests using the already validated asset
   directory. No statistical recalculation or training is needed:

   ```bash
   python paper/build_assets.py --data-dir /path/to/validated-assets \
     --output-dir /path/to/published-assets --repository-commit MERGED_COMMIT_SHA \
     --stage published --tag paper-2026-10-27
   ```

3. Verify `gh auth status`. If necessary, authorize securely with
   `gh auth login --web`. Do not paste tokens into a script or tracked file.
4. Create the tag at that merged commit and publish the four archives and
   `release_manifest.json`:

   ```bash
   gh release create paper-2026-10-27 /path/to/published-assets/tricasrank-paper-*.tar.gz \
     /path/to/published-assets/release_manifest.json \
     --repo ZebraL415/Cas12a-Activity-Ranker --target MERGED_COMMIT_SHA \
     --title 'TriCasRank paper materials (2026-10-27)' \
     --notes-file paper/RELEASE_NOTES.md
   ```

5. Test every asset URL without authentication, unpack into a new directory,
   and run `fetch` and `results --points-only` there. The manifest should refer
   to the merged commit. Confirm the team attachments correspond to this public release.
6. Adopt the publication-ready Availability paragraph and the corrected figure,
   S6 and Table 4 text. The team supplies Acknowledgements and confirms its
   contributors/support. A DOI can be added after registration of an actual
   archival record.

The PR review archives are complete working files with `status=review` and
their base commit recorded; they do not claim to be a published release.
