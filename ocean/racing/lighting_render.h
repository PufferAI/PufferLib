#pragma once

static RenderTexture2D racing_shadow_target(int size) {
    RenderTexture2D target = {};
    target.id=rlLoadFramebuffer();
    Image image=GenImageColor(size,size,WHITE);
    target.texture=LoadTextureFromImage(image);
    UnloadImage(image);
    target.depth.id=rlLoadTextureDepth(size,size,false);
    target.depth.width=target.depth.height=size;
    target.depth.mipmaps=1;
    rlFramebufferAttach(target.id,target.texture.id,RL_ATTACHMENT_COLOR_CHANNEL0,RL_ATTACHMENT_TEXTURE2D,0);
    rlFramebufferAttach(target.id,target.depth.id,RL_ATTACHMENT_DEPTH,RL_ATTACHMENT_TEXTURE2D,0);
    assert(rlFramebufferComplete(target.id));
    SetTextureFilter(target.depth,TEXTURE_FILTER_POINT);
    SetTextureWrap(target.depth,TEXTURE_WRAP_CLAMP);
    return target;
}
