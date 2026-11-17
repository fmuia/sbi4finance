"""Git checkout policy must preserve the tutorial's accepted asset bytes."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from sbi4f_tutorial.trajectories import lesson


class ScientificAssetCheckoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "Git is optional in the participant environment")
    def test_actual_attributes_prevent_autocrlf_smudging_in_an_isolated_repository(self):
        attributes = Path(__file__).resolve().parents[2] / ".gitattributes"
        if not attributes.is_file():
            self.skipTest("checkout policy applies only to the repository distribution")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(attributes, root / ".gitattributes")
            relative = "tutorial/assets/trajectories/rough-known/bundle.json"
            original = (lesson.ASSETS / "trajectories/rough-known/bundle.json").read_bytes().replace(b"\r\n", b"\n")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(original)
            control = b'{"control": true}\n'
            (root / "control.json").write_bytes(control)
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

            def git(*arguments):
                return subprocess.check_output(["git", "-C", str(root), "-c", "core.autocrlf=true", *arguments],
                                               env=env, stderr=subprocess.PIPE)

            git("init", "-q")
            git("add", ".gitattributes", "control.json", relative)
            # Real checkout conversion, without depending on the parent repo's
            # index/staging state or the host's global Git configuration.
            protected = git("cat-file", "--filters", ":" + relative)
            self.assertEqual(protected, original)
            self.assertEqual(git("cat-file", "--filters", ":control.json"), control.replace(b"\n", b"\r\n"))
            self.assertEqual(hashlib.sha256(protected).hexdigest(),
                             lesson.catalogue()["artifacts"]["rough-known"]["bundle_sha256"])


if __name__ == "__main__":
    unittest.main()
