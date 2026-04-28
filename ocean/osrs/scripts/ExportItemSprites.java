/**
 * Export item inventory sprites from OpenRS2 flat cache using RuneLite's
 * ItemSpriteFactory (3D model → 2D sprite rendering).
 *
 * Reads the modern cache at repo-root .refs/osrs-cache-modern/ (OpenRS2 flat format:
 * {index}/{group}.dat files). Renders each requested item ID to a 36x32 PNG
 * matching the real OSRS inventory icon exactly.
 *
 * Usage:
 *   javac -cp <cache-jar>:<deps> scripts/ExportItemSprites.java
 *   java -cp <cache-jar>:<deps>:scripts ExportItemSprites \
 *     --cache .refs/osrs-cache-modern \
 *     --output data/sprites/items \
 *     --ids 4151,10828,21795,...
 */

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

import net.runelite.cache.*;
import net.runelite.cache.definitions.*;
import net.runelite.cache.definitions.loaders.*;
import net.runelite.cache.definitions.providers.*;
import net.runelite.cache.fs.*;
import net.runelite.cache.index.*;
import net.runelite.cache.item.ItemSpriteFactory;

/**
 * Custom Storage that reads OpenRS2 flat cache format ({index}/{group}.dat).
 * Reference tables are at 255/{index}.dat.
 */
class OpenRS2Storage implements Storage {
    private final File cacheDir;

    OpenRS2Storage(File cacheDir) {
        this.cacheDir = cacheDir;
    }

    @Override
    public void init(Store store) throws IOException {
        // discover index IDs from directories under cache root
        // (exclude 255 which is the meta-index)
        File[] dirs = cacheDir.listFiles(File::isDirectory);
        if (dirs == null) throw new IOException("cache dir not readable: " + cacheDir);

        List<Integer> indexIds = new ArrayList<>();
        for (File d : dirs) {
            try {
                int id = Integer.parseInt(d.getName());
                if (id != 255) indexIds.add(id);
            } catch (NumberFormatException ignored) {}
        }
        Collections.sort(indexIds);
        for (int id : indexIds) {
            store.addIndex(id);
        }
    }

    @Override
    public void load(Store store) throws IOException {
        for (Index index : store.getIndexes()) {
            loadIndex(index);
        }
    }

    private void loadIndex(Index index) throws IOException {
        // reference table for this index is at 255/{indexId}.dat
        File refFile = new File(cacheDir, "255/" + index.getId() + ".dat");
        if (!refFile.exists()) {
            System.err.println("warning: no reference table for index " + index.getId());
            return;
        }

        byte[] refData = Files.readAllBytes(refFile.toPath());
        Container container = Container.decompress(refData, null);
        byte[] data = container.data;

        IndexData indexData = new IndexData();
        indexData.load(data);

        index.setProtocol(indexData.getProtocol());
        index.setRevision(indexData.getRevision());
        index.setNamed(indexData.isNamed());

        for (ArchiveData ad : indexData.getArchives()) {
            Archive archive = index.addArchive(ad.getId());
            archive.setNameHash(ad.getNameHash());
            archive.setCrc(ad.getCrc());
            archive.setRevision(ad.getRevision());
            archive.setFileData(ad.getFiles());
        }
    }

    @Override
    public byte[] load(int index, int archive) throws IOException {
        File f = new File(cacheDir, index + "/" + archive + ".dat");
        if (!f.exists()) return null;
        return Files.readAllBytes(f.toPath());
    }

    @Override
    public void save(Store store) throws IOException {
        throw new UnsupportedOperationException("read-only");
    }

    @Override
    public void store(int index, int archive, byte[] data) throws IOException {
        throw new UnsupportedOperationException("read-only");
    }

    @Override
    public void close() throws IOException {}
}

public class ExportItemSprites {
    public static void main(String[] args) throws Exception {
        String cachePath = ".refs/osrs-cache-modern";
        String outputPath = "data/sprites/items";
        String idsArg = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--cache": cachePath = args[++i]; break;
                case "--output": outputPath = args[++i]; break;
                case "--ids": idsArg = args[++i]; break;
            }
        }

        File cacheDir = new File(cachePath);
        File outDir = new File(outputPath);
        outDir.mkdirs();

        System.out.println("opening cache: " + cacheDir.getAbsolutePath());

        try (Store store = new Store(new OpenRS2Storage(cacheDir))) {
            store.load();

            // load items
            ItemManager itemManager = new ItemManager(store);
            itemManager.load();
            itemManager.link();

            // model provider: reads from index 7
            ModelProvider modelProvider = modelId -> {
                Index models = store.getIndex(IndexType.MODELS);
                Archive archive = models.getArchive(modelId);
                if (archive == null) return null;
                byte[] data = archive.decompress(store.getStorage().loadArchive(archive));
                return new ModelLoader().load(modelId, data);
            };

            // sprite manager
            SpriteManager spriteManager = new SpriteManager(store);
            spriteManager.load();

            TextureManager textureManager = new TextureManager(store);
            textureManager.load();

            // parse item IDs to export
            Set<Integer> targetIds = new HashSet<>();
            if (idsArg != null) {
                for (String s : idsArg.split(",")) {
                    targetIds.add(Integer.parseInt(s.trim()));
                }
            }

            int exported = 0, failed = 0;

            Collection<ItemDefinition> items;
            if (targetIds.isEmpty()) {
                // export all items with valid models
                items = itemManager.getItems();
            } else {
                items = new ArrayList<>();
                for (int id : targetIds) {
                    ItemDefinition def = itemManager.getItem(id);
                    if (def != null) ((ArrayList<ItemDefinition>) items).add(def);
                    else System.err.println("  item " + id + ": not found");
                }
            }

            for (ItemDefinition itemDef : items) {
                if (itemDef.name == null || itemDef.name.equalsIgnoreCase("null")) continue;
                if (targetIds.isEmpty() && itemDef.inventoryModel <= 0) continue;

                BufferedImage sprite = ItemSpriteFactory.createSprite(
                    itemManager, modelProvider, spriteManager, textureManager,
                    itemDef.id, 1, 1, 0x303030, false);

                if (sprite == null) {
                    System.err.println("  item " + itemDef.id + " (" + itemDef.name + "): null sprite");
                    failed++;
                    continue;
                }

                File out = new File(outDir, itemDef.id + ".png");
                ImageIO.write(sprite, "PNG", out);
                exported++;

                System.out.println("  " + itemDef.id + " (" + itemDef.name + "): "
                    + sprite.getWidth() + "x" + sprite.getHeight());
            }

            System.out.println("\nexported " + exported + " item sprites, " + failed + " failed");
            System.out.println("output: " + outDir.getAbsolutePath());
        }
    }
}
