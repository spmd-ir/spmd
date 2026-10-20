/* Declarations of the PnetCDF functions and constants miniWeather_mpi.cpp
 * uses, so that it compiles without a PnetCDF installation.  The test builds
 * it with _OUT_FREQ=-1, which turns every call to output() into dead code, so
 * none of these is ever called or linked. */

#ifndef SPMD_EXEC_PNETCDF_STUB_H
#define SPMD_EXEC_PNETCDF_STUB_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int nc_type;

#define NC_NOERR 0
#define NC_WRITE 0x0001
#define NC_CLOBBER 0x0000
#define NC_UNLIMITED 0L
#define NC_DOUBLE 6

int ncmpi_create(MPI_Comm comm, const char *path, int cmode, MPI_Info info,
                 int *ncidp);
int ncmpi_open(MPI_Comm comm, const char *path, int omode, MPI_Info info,
               int *ncidp);
int ncmpi_def_dim(int ncid, const char *name, MPI_Offset len, int *idp);
int ncmpi_def_var(int ncid, const char *name, nc_type xtype, int ndims,
                  const int *dimidsp, int *varidp);
int ncmpi_enddef(int ncid);
int ncmpi_inq_varid(int ncid, const char *name, int *varidp);
int ncmpi_put_vara_double(int ncid, int varid, const MPI_Offset *start,
                          const MPI_Offset *count, const double *op);
int ncmpi_put_vara_double_all(int ncid, int varid, const MPI_Offset *start,
                              const MPI_Offset *count, const double *op);
int ncmpi_begin_indep_data(int ncid);
int ncmpi_end_indep_data(int ncid);
int ncmpi_close(int ncid);
const char *ncmpi_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif
