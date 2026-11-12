"""Synthetic non-87 upload-overlay integrity guards; no simulator is called."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import unittest

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/package_rough_local_repairs_v2.py"
spec=importlib.util.spec_from_file_location("rough_repair_packager_v2",SCRIPT)
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def binding(path):return {"path":str(path),"sha256":package.digest(path)}


class PackageRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.repo=cls.root/"repo";cls.run_root=cls.root/"run";cls.inputs=cls.root/"bundle"
        cls.repo.mkdir();cls.run_root.mkdir();cls.inputs.mkdir()
        kernel=cls.repo/"source/runtime.py";kernel.parent.mkdir();kernel.write_text("# synthetic source-closure fixture; never executed\n")
        h=np.full(63,.001)
        oldplan=cls.inputs/"metadata/old-plan.json"
        write(oldplan,{"tasks":{"joint_unknown":{"design":{"half_spread_forward":h.tolist()}}},"numerical_policy":{"paths_ladder":[32768,65536]}})
        oldplan_sha=package.digest(oldplan);bundle=cls.inputs/"bundle.sqlite"
        db=sqlite3.connect(bundle)
        db.executescript("CREATE TABLE attempts(source_key,id,block,rung,status,summary,payload); CREATE TABLE draws(source_key,id,value); CREATE TABLE rows(source_key,id,status,summary,payload);")
        rows=[];records=[]
        for index in range(3):
            rowid=1000+index;source=f"master:{index:06d}"
            physical=np.array([.1,1.,-.5,.04,.03,2.])+index*1e-6;c=1.2
            prior={"parameters":physical.tolist(),"noise_scale":c,"identity_entropy":[1380073777,20261111,0,rowid]}
            identity={"repair_plan_sha256":"pending","row_index":index,"original_identity":{"plan_sha256":oldplan_sha,"job":{"lane":"joint_unknown","split":"train","start":rowid,"stop":rowid+1}},
                "original_row_id":rowid,"original_failure":{"spec":prior},"original_split":"train","producer_kind":"overnight"}
            item={"plan_sha256":oldplan_sha,"bank_lane":"joint_unknown","split":"train","id":rowid,"identity":identity["original_identity"],"producer_kind":"overnight","source_key":source,"source_journal":f"/original/shard-{index:06d}/journal.sqlite"}
            rows.append(item);records.append(identity)
            summary=json.dumps({"paths":32768,"entropy":[*prior["identity_entropy"],1,1,0,0]})
            blob=f"unchanged-original-block-{index}".encode()
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?)",(source,rowid,0,0,"accepted",summary,blob))
            db.execute("INSERT INTO draws VALUES(?,?,?)",(source,rowid,json.dumps(prior)))
            db.execute("INSERT INTO rows VALUES(?,?,?,?,NULL)",(source,rowid,"failed",json.dumps({"spec":prior})))
        db.commit();db.close()
        transport=cls.inputs/"manifest.json"
        write(transport,{"bundle":{"path":"bundle.sqlite","sha256":package.digest(bundle)},"producers":{"overnight":{"plan":{"path":"metadata/old-plan.json","sha256":oldplan_sha}}}})
        archive=cls.root/"input-bundle.tar.gz";archive.write_bytes(b"synthetic original transport identity")
        plan={"rows":rows,"transport_manifest":binding(transport),"transport_archive":binding(archive),
            "transport_files":{str(path):package.digest(path) for path in (oldplan,bundle,transport,archive)},
            "source_bindings":{"source/runtime.py":package.digest(kernel)},"join_rule":"replace matching original failed key only",
            "cross_platform_scope":"synthetic fixture; no pricing"}
        write(cls.run_root/"plan.json",plan);plan_sha=package.digest(cls.run_root/"plan.json")
        cls.row_hashes={}
        original=sqlite3.connect(bundle)
        for index,(item,identity) in enumerate(zip(rows,records)):
            directory=cls.run_root/"output/rows"/f"row-{index:03d}";directory.mkdir(parents=True)
            identity["repair_plan_sha256"]=plan_sha
            p=directory/"data.npz";phi=np.asarray(identity["original_failure"]["spec"]["parameters"])
            prices=np.full(63,.1+index*1e-4);z=np.linspace(-1,1,63);c=1.2
            np.savez_compressed(p,parameters=np.r_[phi,c],physical_parameters=phi,noise_scale=np.asarray(c),
                reference_prices_forward=prices,observations_forward=prices+c*h*z,quote_standard_normals=z,
                pricing_covariance_forward=np.eye(63)*1e-10,refinement_evaluated=np.asarray(False),continuous_bias_certified=np.asarray(False))
            j=directory/"journal.sqlite";local=sqlite3.connect(j)
            local.executescript("CREATE TABLE identity(value); CREATE TABLE attempts(block,rung,status,summary,payload,origin);")
            local.execute("INSERT INTO identity VALUES(?)",(json.dumps(identity),))
            entry=original.execute("SELECT block,rung,status,summary,payload FROM attempts WHERE source_key=?",(item["source_key"],)).fetchone()
            local.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)",(*entry,"oscar_reused"))
            for block in range(1,7):
                summary=json.dumps({"paths":32768,"entropy":[*identity["original_failure"]["spec"]["identity_entropy"],1,1,block,0]})
                local.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)",(block,0,"accepted",summary,b"local fixture","local_recomputed"))
            local.commit();local.close()
            write(directory/"manifest.json",{"status":"complete","identity":identity,"payload":binding(p),"journal":binding(j),"SBI_training_readiness":False,"scientific_acceptance":False})
            cls.row_hashes[index]=package.digest(p)
        original.close()
        write(cls.run_root/"output/batch-status.json",{"status":"finished","requested":3,"pending":[],"active":[]})
        cls.checker=cls.root/"independent-check.py";cls.checker.write_text("# mocked independent audit fixture; no MC\n")
        cls.audit=cls.root/"audit.json"
        write(cls.audit,{"format":"rough_actual63_independent_local_repair_payload_check_v1","complete":True,"complete_count":3,
            "requested_indices":list(range(3)),"missing_indices":[],"failed_indices":[],"prices_generated":0,
            "original_transport_hashes_unchanged":True,"repair_plan":{"sha256":plan_sha},"checker_sha256":package.digest(cls.checker),
            "rows":[{"index":i,"payload_sha256":digest} for i,digest in cls.row_hashes.items()]})
        cls.output=cls.root/"packed"
        cls.receipt=package.package(cls.repo,cls.run_root,cls.output,cls.audit,cls.checker)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def copy_package(self,name):
        target=self.root/name
        if target.exists():shutil.rmtree(target)
        shutil.copytree(self.output,target);return target

    @staticmethod
    def refresh_inventory(tree):
        manifest=package.read(tree/"package-manifest.json")
        for relative in manifest["files"]:
            path=tree/relative
            manifest["files"][relative]={"sha256":package.digest(path),"bytes":path.stat().st_size}
        write(tree/"package-manifest.json",manifest)

    def test_roundtrip_relocation_preserves3_unique_rows_sources_and_reused_bytes(self):
        result=package.validate_tree(self.output)
        self.assertEqual(result["rows_verified"],3);self.assertEqual(result["unique_original_join_keys"],3)
        self.assertEqual(result["reused_attempts_byte_verified"],3);self.assertEqual(result["local_recomputed_attempts"],18)
        destination=self.root/"relocated"
        result=package.validate_upload(Path(self.receipt["archive"]),self.receipt["archive_sha256"],destination)
        self.assertEqual(result["rows_verified"],3);self.assertFalse(result["original_production_payloads_modified"])

    def test_changed_payload_and_source_are_rejected(self):
        for name,path in (("tampered-payload","repair/output/rows/row-000/data.npz"),("tampered-source","sources/source/runtime.py")):
            tree=self.copy_package(name)
            with (tree/path).open("ab") as stream:stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"content changed"):package.validate_tree(tree)

    def test_duplicate_join_and_truncated_rows_are_rejected(self):
        for name,change in (("bad-join",lambda m:m["rows"][1].update(join_key=m["rows"][0]["join_key"])),("truncated",lambda m:m["rows"].pop())):
            tree=self.copy_package(name);manifest=package.read(tree/"package-manifest.json");change(manifest);write(tree/"package-manifest.json",manifest)
            with self.assertRaises(ValueError):package.validate_tree(tree)

    def test_immutable_relocation_and_source_hashes_survive_inventory_rewriting(self):
        tree=self.copy_package("rebound-input")
        with (tree/"inputs/bundle/metadata/old-plan.json").open("a") as stream:stream.write(" ")
        self.refresh_inventory(tree)
        with self.assertRaisesRegex(ValueError,"binding|relocation"):package.validate_tree(tree)
        tree=self.copy_package("rebound-source")
        (tree/"sources/source/runtime.py").write_text("# changed financial-source fixture\n")
        self.refresh_inventory(tree)
        with self.assertRaisesRegex(ValueError,"source closure"):package.validate_tree(tree)
        tree=self.copy_package("rebound-audit-checker")
        (tree/"audit/independent-check.py").write_text("# substituted independent audit checker\n")
        self.refresh_inventory(tree)
        with self.assertRaisesRegex(ValueError,"checker"):package.validate_tree(tree)

    def test_reused_attempt_must_match_original_even_if_local_journal_rehashed(self):
        tree=self.copy_package("rebound-reused-block")
        row_manifest=tree/"repair/output/rows/row-000/manifest.json";record=package.read(row_manifest)
        relative="repair/output/rows/row-000/journal.sqlite";journal=tree/relative
        with sqlite3.connect(journal) as db:db.execute("UPDATE attempts SET payload=? WHERE origin='oscar_reused'",(b"changed reused block",));db.commit()
        record["journal"]["sha256"]=package.digest(journal);write(row_manifest,record)
        relocation=package.read(tree/"relocation.json")
        relocation["direct_files"][record["journal"]["path"]]["sha256"]=package.digest(journal)
        write(tree/"relocation.json",relocation);self.refresh_inventory(tree)
        with self.assertRaisesRegex(ValueError,"reused original attempt"):package.validate_tree(tree)

    def test_missing_audit_duplicate_audit_and_incomplete_batch_refused(self):
        audit=package.read(self.audit);audit["rows"].append(audit["rows"][0])
        with self.assertRaises(ValueError):package.audit_check(audit,package.digest(self.run_root/"plan.json"),self.row_hashes)
        batch=self.run_root/"output/batch-status.json";original=batch.read_bytes()
        try:
            write(batch,{"status":"running","requested":3,"pending":[2],"active":[]})
            with self.assertRaisesRegex(ValueError,"finished"):package.package(self.repo,self.run_root,self.root/"incomplete",self.audit,self.checker)
        finally:batch.write_bytes(original)
        with self.assertRaises(FileNotFoundError):package.package(self.repo,self.run_root,self.root/"missing-audit",self.root/"absent-audit.json",self.checker)


    def test_audit_count_is_dynamic_and_matches_exact_indices(self):
        audit=package.read(self.audit)
        audit.update(complete_count=2,requested_indices=[0,1],rows=audit["rows"][:2])
        package.audit_check(audit,package.digest(self.run_root/"plan.json"),{i:self.row_hashes[i] for i in range(2)})
        with self.assertRaisesRegex(ValueError,"positive contiguous"):
            package.audit_check(audit,package.digest(self.run_root/"plan.json"),{})
        with self.assertRaises(ValueError):
            package.audit_check(audit,package.digest(self.run_root/"plan.json"),self.row_hashes)
        tree=self.copy_package("bad-declared-count")
        manifest=package.read(tree/"package-manifest.json");manifest["row_count"]=2
        write(tree/"package-manifest.json",manifest)
        with self.assertRaisesRegex(ValueError,"declared"):
            package.validate_tree(tree)


class UploadArchiveSafetyTests(unittest.TestCase):
    def test_traversal_links_and_duplicate_members_refused_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for case in ("traversal","absolute","symlink","duplicate"):
                archive=root/(case+".tar")
                with tarfile.open(archive,"w") as tar:
                    names=["../escaped"] if case=="traversal" else ["/absolute"] if case=="absolute" else ["same","same"] if case=="duplicate" else ["linked"]
                    for name in names:
                        member=tarfile.TarInfo(name)
                        if case=="symlink":member.type=tarfile.SYMTYPE;member.linkname="/outside";tar.addfile(member)
                        else:member.size=1;tar.addfile(member,io.BytesIO(b"x"))
                with self.assertRaises(ValueError):package.validate_upload(archive,package.digest(archive),root/(case+"-out"))
            self.assertFalse((root/"escaped").exists())


if __name__=="__main__":unittest.main()
