package com.hophacks.guidedog;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.View;

import androidx.core.content.res.ResourcesCompat;

public class Draw extends View {
    Paint borderPaint, textPaint;
    Rect rect;
    String text;
    int offset_X;
    int offset_Y = 150;

    public Draw(Context context, Rect rect, String text) {
        super(context);
        this.rect = rect;
        this.text = text;

        borderPaint = new Paint();
        borderPaint.setColor(Color.WHITE);
        borderPaint.setStrokeWidth(5f);
        borderPaint.setStyle(Paint.Style.STROKE);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setStrokeWidth(50f);
        textPaint.setTextSize(32f);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTypeface(ResourcesCompat.getFont(context, R.font.josefin_sans_regular));
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawText(text, rect.left, rect.top+offset_Y-20, textPaint);
        canvas.drawRect(rect.left, rect.top+offset_Y, rect.right, rect.bottom+offset_Y, borderPaint);
    }

}
