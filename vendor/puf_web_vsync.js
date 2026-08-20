addToLibrary({
  puf_web_vsync__async: true,
  puf_web_vsync: () => Asyncify.handleSleep((wakeUp) => {
    requestAnimationFrame(() => wakeUp());
  }),
});
