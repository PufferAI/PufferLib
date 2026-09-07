#ifndef FC_ASSETS_H
#define FC_ASSETS_H

#include <stddef.h>
#include <stdio.h>

#define FC_ASSET_PATH_MAX 1024

const char* fc_asset_root(void);
const char* fc_repo_root(void);

int fc_asset_resolve_path(const char* logical_path, char* out, size_t cap);
int fc_repo_resolve_path(const char* logical_path, char* out, size_t cap);
int fc_asset_exists(const char* logical_path);

FILE* fc_asset_fopen(const char* logical_path, const char* mode);
int fc_asset_close(FILE* f);
unsigned char* fc_asset_read_all(const char* logical_path, size_t* out_size);

#endif /* FC_ASSETS_H */
