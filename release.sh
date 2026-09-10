#!/usr/bin/env bash
# Cut a release of velend.
#
#	./release.sh v0.3.0                 # every platform
#	./release.sh v0.3.0 macos-arm64     # only the platforms named
#
# Sets the version in the manifest, builds the extension zips into build/, then
# amends the manifest into HEAD and tags it. The build runs before the amend so
# that a build failure leaves nothing behind, and so that the wheel and platform
# lists scripts/build.py regenerates land in the tagged commit too.
#
# Amending rewrites HEAD. If HEAD is already on the remote, the push at the end
# needs --force-with-lease; the script says so when it notices.
#
# Nothing is pushed. Push the branch and the tag yourself once the zips look right.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$ROOT/blender_manifest.toml"

fail() {
	echo "error: $*" >&2
	exit 1
}

[ $# -ge 1 ] || fail "usage: $(basename "$0") vMAJOR.MINOR.PATCH [PLATFORM...]"

TAG="$1"
shift
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "tag should look like v0.3.0, got '$TAG'"
VERSION="${TAG#v}"

cd "$ROOT"

# A release has to be reproducible from the tag, so refuse to build one on top
# of edits that are not in the commit being tagged.
git diff-index --quiet HEAD -- || fail "working tree has uncommitted changes"
! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || fail "tag $TAG already exists"

# Amending a commit that is already published rewrites history someone else may
# have. Not a refusal -- it is your branch -- but it should not be a surprise.
UPSTREAM="$(git rev-parse -q --verify --abbrev-ref '@{upstream}' 2>/dev/null || true)"
PUSHED=0
if [ -n "$UPSTREAM" ] && git merge-base --is-ancestor HEAD "$UPSTREAM"; then
	PUSHED=1
fi

CURRENT="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$MANIFEST")"
[ -n "$CURRENT" ] || fail "no version in $MANIFEST"

echo "== releasing $CURRENT -> $VERSION as $TAG"

if [ "$PUSHED" -eq 1 ]; then
	echo "   HEAD is already on $UPSTREAM; amending it rewrites published history"
fi

# Until the amend lands, any exit undoes the version bump and whatever
# scripts/build.py rewrote in the manifest.
amended=0
restore() {
	if [ "$amended" -eq 0 ]; then
		git checkout -- "$MANIFEST"
	fi
}
trap restore EXIT

# Not `sed -i`, whose spelling differs between BSD and GNU.
sed "s/^version = \".*\"$/version = \"$VERSION\"/" "$MANIFEST" > "$MANIFEST.new"
mv "$MANIFEST.new" "$MANIFEST"
grep -q "^version = \"$VERSION\"\$" "$MANIFEST" || fail "failed to set the version in $MANIFEST"

echo
python3 scripts/build.py "$@"

echo
echo "== amending HEAD and tagging"
git add "$MANIFEST"
# --no-edit, because the commit keeps the message it already has.
git commit --amend --no-edit
amended=1
git tag -a "$TAG" -m "velend $TAG"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo
echo "tagged $TAG at $(git rev-parse --short HEAD)"
if [ "$PUSHED" -eq 1 ]; then
	echo "push it with: git push --force-with-lease origin $BRANCH && git push origin $TAG"
else
	echo "push it with: git push origin $BRANCH $TAG"
fi
