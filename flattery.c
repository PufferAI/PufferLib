// Updated by PufferAI from the old flattery library
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <ctype.h> // for isdigit()
#include <stdlib.h> // for atol

static PyObject *
unflatten(PyObject *ignore, PyObject *args)
{
  PyObject *src = NULL;
  PyObject *dst = NULL;
  PyObject *nonelist = NULL;
  PyObject *slot = NULL;
  PyObject *slotvalue = NULL;
  PyObject *part = NULL;

  if (!PyArg_ParseTuple(args, "O!:unflatten", &PyDict_Type, &src))
    return NULL;

  /* Create a [None] list. Used for extending lists to higher indices. */

  if (!(nonelist = PyList_New(0)))
    goto error;
  if (PyList_Append(nonelist, Py_None) < 0)
    goto error;

  /* Iterate through key value pairs in the src dict,
     building the nested data structure in dst as we go. */

  PyObject *k, *v;
  Py_ssize_t pos = 0;

  while (PyDict_Next(src, &pos, &k, &v))
  {
    const char *key = PyUnicode_AsUTF8(k);
    const char *p;

    p = key;

    if (dst == NULL) { // Check if dst is not initialized
      char type_char = *p; // Extract the type character for the first key
      if (type_char == 'T') {
        dst = PyList_New(0); // We'll convert it to a tuple later
      } else if (type_char == 'L') {
        dst = PyList_New(0);
      } else if (type_char == 'D') {
        dst = PyDict_New();
      } else if (type_char == 'V') {
        dst = v;
        Py_INCREF(dst);
        Py_DECREF(nonelist);
        return dst; // If the entire object is a value, return it directly
      } else {
        goto error;
      }
      if (dst == NULL) goto error;
    }

    slot = dst;
    Py_INCREF(slot);

    do
    {
      /* Extract current part of the key path. */

      const char *start = p;
      while (*p && *p != '.') p++;

      /* Determine the type from the prefix and remove it. */
      start += 1; // Skip the type char
      part = PyUnicode_FromStringAndSize(start, p-start);

      /* Advance to next part of key path, unless at the end. */

      if (*p == '.')
        p++;
    
      /* What value should we insert under this slot?
         - if this is the last path part, insert the value from src.
         - if the next path part is numeric, insert an empty list.
         - otherwise, insert an empty hash.
       */

      if (*p == 'V') {
        p++;
        slotvalue = v;
        Py_INCREF(slotvalue);
      }
      else if (*p == 'T') {
        slotvalue = PyList_New(0); // We'll convert it to a tuple later
      }
      else if (*p == 'L') {
        slotvalue = PyList_New(0);
      }
      else if (*p == 'D') {
        slotvalue = PyDict_New();
      }
      else {
        goto error;
      }

      if (!slotvalue)
        goto error;

      /* Assign to the current slot. */

      if (isdigit(*start))
      {
        /* If the current path part is numeric, index into a list. */

        if (!PyList_Check(slot))
          goto error;

        // FIXME thorough error checking here

        Py_ssize_t len = PyList_Size(slot);
        Py_ssize_t index = atol(PyUnicode_AsUTF8(part));

        /* Extend the list with [None,None,...] if necessary. */

        if (index >= len)
        {
          PyObject *tail = PySequence_Repeat(nonelist, index-len+1);
          PyObject *extended = PySequence_InPlaceConcat(slot, tail);
          Py_DECREF(tail);
          Py_DECREF(extended);
        }

        /* Don't clobber an existing entry.
           Caveat: PyList_SetItem() steals a reference to slotvalue. */

        PyObject *extant = NULL;

        if ((extant = PyList_GetItem(slot, index)) == Py_None) {
          PyList_SetItem(slot, index, slotvalue);
          Py_INCREF(slotvalue);
        }
        else {
          Py_DECREF(slotvalue);
          slotvalue = extant;
          Py_INCREF(slotvalue);
        }
      }
      else
      {
        /* If the current path part is non-numeric, index into a dict. */

        if (!PyDict_Check(slot))
          goto error;

        /* Don't clobber an existing entry. */

        PyObject *extant = NULL;

        if (!(extant = PyDict_GetItem(slot, part)))
          PyDict_SetItem(slot, part, slotvalue);
        else {
          Py_DECREF(slotvalue);
          slotvalue = extant;
          Py_INCREF(slotvalue);
        }

      }

      /* Descend further into the dst data structure. */

      Py_DECREF(slot);
      slot = slotvalue;
      slotvalue = NULL;

      Py_DECREF(part);
      part = NULL;
    }
    while (*p);

    Py_DECREF(slot);
    slot = NULL;
  }

  Py_DECREF(nonelist);
  return dst;

error:

  Py_XDECREF(dst);
  Py_XDECREF(nonelist);
  Py_XDECREF(slot);
  Py_XDECREF(slotvalue);
  Py_XDECREF(part);

  return NULL;
}

static PyObject *
flatten_internal(PyObject *src)
{
  PyObject *flat = NULL;
  PyObject *dst = NULL;

  if (PyTuple_Check(src))
  {
    if (!(flat = PyDict_New()))
      goto error;

    Py_ssize_t i;
    Py_ssize_t len = PyTuple_Size(src);

    for (i=0; i<len; i++)
    {
      PyObject *elem = PyTuple_GetItem(src,i);
      if (elem == Py_None && i<len-1) continue;
      Py_INCREF(elem);
      PyObject *o = flatten_internal(elem);
      Py_DECREF(elem);
      PyObject *k = PyUnicode_FromFormat("T%zd", i); // T for tuple
      PyDict_SetItem(flat, k, o);
      Py_DECREF(k);
      Py_DECREF(o);
    }
  }
  else if (PyList_Check(src))
  {
    if (!(flat = PyDict_New()))
      goto error;

    /* Iterate through elements in the list src, recursively flattening.
       Skip any entries which are None -- use a sparse encoding. */

    Py_ssize_t i;
    Py_ssize_t len = PyList_Size(src);

    for (i=0; i<len; i++)
    {
      PyObject *elem = PyList_GetItem(src,i);
      if (elem == Py_None && i<len-1) continue;
      Py_INCREF(elem);
      PyObject *o = flatten_internal(elem);
      Py_DECREF(elem);
      PyObject *k = PyUnicode_FromFormat("L%zd", i); // L for list
      PyDict_SetItem(flat, k, o);
      Py_DECREF(k);
      Py_DECREF(o);
    }
  }
  else if (PyDict_Check(src))
  {
    if (!(flat = PyDict_New()))
      goto error;

    /* Iterate through pairs in the dict src, recursively flattening. */

    PyObject *k, *v;
    Py_ssize_t pos = 0;

    while (PyDict_Next(src, &pos, &k, &v))
    {
      Py_INCREF(v);
      PyObject *o = flatten_internal(v);
      Py_DECREF(v);
      PyObject *k_with_prefix = PyUnicode_FromFormat("D%U", k); // D for dict
      PyDict_SetItem(flat, k_with_prefix, o);
      Py_DECREF(o);
    }
  }
  else
  {
    /* The Python object is a scalar or something we don't know how
       to flatten, return it as-is. */

    if (!(dst = PyDict_New()))
      goto error;

    PyObject *key = PyUnicode_FromString("V"); // V for value
    PyDict_SetItem(dst, key, src);
    Py_DECREF(key);
    Py_INCREF(src);
    return dst;
  }

  /* Roll up recursively flattened dictionaries. */

  if (!(dst = PyDict_New()))
    goto error;

  PyObject *k1, *v1;
  Py_ssize_t pos1 = 0;

  while (PyDict_Next(flat, &pos1, &k1, &v1))
  {
    if (PyDict_Check(v1))
    {
      PyObject *k2, *v2;
      Py_ssize_t pos2 = 0;

      while (PyDict_Next(v1, &pos2, &k2, &v2))
      {
        const char *k1c = PyUnicode_AsUTF8(k1);
        const char *k2c = PyUnicode_AsUTF8(k2);
        PyObject *k = PyUnicode_FromFormat("%s.%s",k1c,k2c);
        PyDict_SetItem(dst, k, v2);
        Py_DECREF(k);
      }
    }
    else
      PyDict_SetItem(dst, k1, v1);
  }

  Py_DECREF(flat);

  return dst;

error:

  Py_XDECREF(dst);
  Py_XDECREF(flat);

  return NULL;
}

static PyObject *
flatten(PyObject *ignore, PyObject *args)
{
  PyObject *src = NULL;

  if (!PyArg_ParseTuple(args, "O:flatten", &src))
    return NULL;

  return flatten_internal(src);
}

/* List of free functions defined in the module */

static PyMethodDef flattery_methods[] = {
  {"unflatten", unflatten, METH_VARARGS, "unflatten(dict) -> dict"},
  {"flatten", flatten, METH_VARARGS, "flatten(dict) -> dict"},
  {NULL, NULL}    /* sentinel */
};

PyDoc_STRVAR(module_doc, "Flattery: fast flattening and unflattening of nested data structures.");

/* Initialization function for the module */

static struct PyModuleDef cextmodule = {
  PyModuleDef_HEAD_INIT,
  "flattery.cext",
  module_doc,
  -1,
  flattery_methods
};

PyMODINIT_FUNC
PyInit_cext(void)
{
  return PyModule_Create(&cextmodule);
}