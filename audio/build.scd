// Offline SynthDef compiler.
//
// Compiles every audio/synthdefs/*.scd in this directory to a .scsyndef binary
// in web/synthdefs/. Does NOT require a running scsynth server (writeDefFile
// serialises the def to disk directly). See audio/README.md for how to run it.

(
	var here = thisProcess.nowExecutingPath.dirname;
	var srcDir = here +/+ "synthdefs";
	~outDir = (here +/+ ".." +/+ "web" +/+ "synthdefs").standardizePath;

	File.mkdir(~outDir);

	PathName(srcDir).files.do({ |f|
		if (f.extension == "scd") {
			("compiling: " ++ f.fileName).postln;
			f.fullPath.load;
		};
	});

	("wrote .scsyndef files to " ++ ~outDir).postln;
	0.exit;
)
