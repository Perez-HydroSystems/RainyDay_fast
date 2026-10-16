# RainyDay_mod (Perez-HydroSystems)

> **About this fork.** This is a modified clone of the original RainyDay repository
> (Daniel B. Wright and the Hydroclimate Extremes Group, https://github.com/HydroclimateExtremesGroup/RainyDay).
> It contains many changes by Gabriel Perez (GP) that are **not** in the original code, including:
> - batched multithreaded CPU / process-pool / GPU (TensorFlow-Metal, CuPy-CUDA) engines for the storm-catalog scan,
>   reader-thread prefetching, and an optional JSON `COMPUTE` block (see `CLAUDE.md`);
> - `CATALOGPATH` to reuse an existing storm catalog from another folder, with consistency warnings;
> - transposition on irregular domains now requires the **whole watershed inside the domain**, the same rule as the
>   storm-catalog search (GP 2026-10-16, see the comments in `Source/RainyDay_Py3.py`);
> - several bug fixes and diagnostic-plot changes.
>
> Changes are marked in the code with `GP` comments; previous implementations are kept as comments.
> Results can differ from the original RainyDay. The original description follows.

---

Welcome to RainyDay. RainyDay is a framework for generating large numbers of realistic extreme rainfall scenarios based on relatively short records of remotely-sensed precipitation fields.  It is founded on a statistical resampling concept known as stochastic storm transposition (SST).  These rainfall scenarios can then be used to examine the extreme rainfall statistics for a user-specified region, or to drive a hazard model (usually a hydrologic model, but the method produces output that would also be useful for landslide models). RainyDay is well suited to flood modeling in small and medium-sized watersheds.  The framework is made to be simple yet powerful and easily modified to meet specific needs, taking advantage of Python’s simple syntax and well-developed libraries.  It is still a work in progress.  Therefore, the contents of the guide may be out-of-date.

Development of RainyDay has been supported by NASA, the National Science Foundation, The Wisconsin Alumni Research Foundation, and the Research and Development Office at the U.S. Bureau of Reclamation.

The latest version of RainyDay is distributed under the MIT open source license: https://opensource.org/licenses/MIT

