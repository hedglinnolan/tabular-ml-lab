/** A small pretend file system for GET /api/fs/list in mock mode. */
import type { FsListing } from "../api/schema";
import { dietaryRecalls, genomicsWide, type MockDataset } from "./datasets";

interface Node {
  size?: number;
  children?: Record<string, Node>;
  dataset?: () => MockDataset;
}

const HOME = "/Users/researcher";

const TREE: Node = {
  children: {
    Users: {
      children: {
        researcher: {
          children: {
            data: {
              children: {
                "dietary_recalls.csv": { size: 58_214, dataset: dietaryRecalls },
                "genomics_counts_wide.csv": { size: 612_880, dataset: () => genomicsWide() },
                "codebook.pdf": { size: 184_320 },
                "README.txt": { size: 2_048 },
                nhanes: {
                  children: {
                    "dietary_recalls_2017.parquet": { size: 41_902, dataset: dietaryRecalls },
                  },
                },
                omics: {
                  children: {
                    "counts_batch1.tsv": { size: 598_311, dataset: () => genomicsWide() },
                  },
                },
              },
            },
            Documents: { children: {} },
          },
        },
      },
    },
  },
};

function lookup(path: string): Node | null {
  const parts = path.split("/").filter(Boolean);
  let node: Node | undefined = TREE;
  for (const p of parts) {
    node = node?.children?.[p];
    if (!node) return null;
  }
  return node ?? null;
}

export function listDir(path: string | null): FsListing | null {
  const target = path && path.length ? path.replace(/\/+$/, "") || "/" : HOME;
  const node = lookup(target);
  if (!node?.children) return null;
  const parentParts = target.split("/").filter(Boolean).slice(0, -1);
  return {
    path: target,
    parent: target === "/" ? null : "/" + parentParts.join("/"),
    entries: Object.entries(node.children)
      .map(([name, child]) => ({
        name,
        path: `${target === "/" ? "" : target}/${name}`,
        is_dir: Boolean(child.children),
        size: child.children ? null : (child.size ?? null),
      }))
      .sort((a, b) => Number(b.is_dir) - Number(a.is_dir) || a.name.localeCompare(b.name)),
  };
}

export function datasetAt(path: string): MockDataset | null {
  const node = lookup(path);
  if (!node?.dataset) return null;
  const ds = node.dataset();
  return { ...ds, name: path.split("/").pop() ?? ds.name };
}

export const MOCK_HOME = HOME;
