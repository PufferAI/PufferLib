#pragma once

static Font racing_hud_font;
static void racing_hud_text(const char *text,int x,int y,int size,Color color) {
    DrawTextEx(racing_hud_font,text,(Vector2){(float)x,(float)y},(float)size,0.5f,color);
}
static float racing_hud_measure(const char *text,int size) {
    return MeasureTextEx(racing_hud_font,text,(float)size,0.5f).x;
}
static const char *racing_lap_text(double seconds) {
    return seconds > 0 ? TextFormat("%d:%06.3f",(int)seconds/60,fmod(seconds,60)) : "--:--.---";
}
static const char *racing_car_status(const RacingCarFrame &f) {
    if (f.task_done == 7) return "FINISHED";
    if (f.task_done == 5) return "DNF / STALLED";
    if (f.task_done == 6) return "DNF / TIMEOUT";
    if (f.task_done || f.crashed) return "DNF / CRASH";
    return "RACING";
}
static void racing_hud_draw(const RacingCarFrame *frame, bool leader_follow,
        bool manual, bool diagnostics, bool lidar) {
    int width=GetScreenWidth(), height=GetScreenHeight();
    auto &g=racing_ghosts;
    const char *name=g.count && g.selected>=0 ? racing_ghost_names[g.selected] : "Race overview";
    Color accent=g.count && g.selected>=0 ? racing_ghost_colors[g.selected] : SKYBLUE;
    DrawRectangle(0,0,width,38,Fade((Color){12,16,23,255},0.92f));
    DrawRectangle(0,0,5,38,accent);
    racing_hud_text(name,18,10,20,WHITE);
    racing_hud_text(leader_follow ? "AUTO / LEADER" : manual ? "MANUAL" : "POLICY",
        width-150,12,14,accent);
    if (g.count) {
        int order[RACING_MAX_GHOSTS];
        for (int i=0;i<g.count;++i) order[i]=i;
        std::stable_sort(order,order+g.count,[&](int a,int b) {
#ifdef RACING_MULTI
            return g.rank[a]<g.rank[b];
#else
            return g.best_progress[a]>g.best_progress[b];
#endif
        });
        int panel=width<900 ? 240 : 290;
        DrawRectangle(12,52,panel,35+g.count*27,Fade((Color){12,16,23,255},0.85f));
        racing_hud_text("POS   DRIVER",24,63,13,LIGHTGRAY);
        racing_hud_text("GAP TO P1",12+panel-108,63,13,LIGHTGRAY);
        for (int row=0;row<g.count;++row) {
            int i=order[row], y=88+27*row;
            const RacingCarFrame &f=g.frames[i];
            if (i==g.selected) DrawRectangle(16,y-4,panel-8,25,Fade(accent,0.12f));
            racing_hud_text(TextFormat("%d",row+1),24,y,15,racing_ghost_colors[i]);
            // Fit names rather than allowing text to collide with the results column.
            char label[64]; snprintf(label,sizeof(label),"%s",racing_ghost_names[i]);
            int available=panel-151;
            while (strlen(label)>1 && racing_hud_measure(label,14)>available) label[strlen(label)-1]=0;
            racing_hud_text(label,49,y,14,racing_ghost_colors[i]);
            const char *result;
            if (f.task_done==7) result=racing_lap_text(f.last_lap);
            else if (f.task_done || f.crashed) result=f.task_done==5 ? "DNF / STALL" : f.task_done==6 ? "TIMEOUT" : "DNF";
            else if (row==0) result="LEADER";
            // Gaps use race-leader route distance, rather than distance to the preceding car.
            else result=TextFormat("+%.0f m",fmaxf(0,g.frames[order[0]].progress-f.progress));
            racing_hud_text(result,12+panel-108,y,13, f.task_done && f.task_done!=7 ? GRAY : RAYWHITE);
        }
    }
    if (frame && (!g.count || g.selected>=0)) {
        int y=height-157;
        DrawRectangle(12,y,330,116,Fade((Color){12,16,23,255},0.88f));
        racing_hud_text(TextFormat("%.0f",frame->speed*3.6f),24,y+10,38,WHITE);
        racing_hud_text(TextFormat("km/h    G%d",frame->gear),109,y+25,17,accent);
        racing_hud_text("CURRENT",24,y+55,11,GRAY);
        racing_hud_text(racing_lap_text(frame->lap_time),24,y+72,17,WHITE);
        racing_hud_text("LAST",132,y+55,11,GRAY);
        racing_hud_text(racing_lap_text(frame->last_lap),132,y+72,17,WHITE);
        racing_hud_text("BEST",240,y+55,11,GRAY);
        racing_hud_text(racing_lap_text(frame->best_lap),240,y+72,17,accent);
        DrawRectangle(24,y+101,137,4,Fade(GREEN,0.18f));
        DrawRectangle(24,y+101,(int)(137*Clamp(frame->throttle,0,1)),4,GREEN);
        DrawRectangle(177,y+101,137,4,Fade(RED,0.18f));
        DrawRectangle(177,y+101,(int)(137*Clamp(frame->brake,0,1)),4,RED);
        if (diagnostics) {
            DrawRectangle(12,y-78,360,66,Fade(BLACK,0.8f));
            racing_hud_text(TextFormat("%s | reward %.3f | %.0f m",racing_car_status(*frame),
                frame->reward,frame->progress),23,y-66,14,WHITE);
            racing_hud_text(TextFormat("Grip %.2f %.2f %.2f %.2f | steer %.2f",
                frame->grip[0],frame->grip[1],frame->grip[2],frame->grip[3],frame->steering),23,y-44,13,LIGHTGRAY);
        }
    } else racing_hud_text("No active cars / waiting for the next race",18,height-68,18,LIGHTGRAY);
    DrawRectangle(0,height-30,width,30,Fade((Color){12,16,23,255},0.75f));
    racing_hud_text(TextFormat("1/2/3 views   4 overview   C leader   [ ] car   L LiDAR%s   Tab control   F1 HUD   F2 details",
        lidar ? " ON" : ""),18,height-23,width<1000 ? 11 : 13,GRAY);
}
