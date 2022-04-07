from cProfile import label
import tkinter as tki
from tkinter.font import BOLD


GRID_GAP = (8, 8)


class UI:
    def get_label(title: str, row=0, column=0) -> tki.Label:
        def make(window):
            label = tki.Label(window, text=title)
            label.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return label

        return make

    def get_title(title: str, row=0, column=0) -> tki.Label:
        def make(window):
            label = tki.Label(window, text=title,
                              font=("Helvetica", 24, BOLD))
            label.grid(row=row, column=column,
                       padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return label

        return make

    def get_button(text: str, on_click, row=0, column=0, width=10) -> tki.Button:
        def make(window):
            button = tki.Button(window, text=text,
                                width=width, command=on_click, highlightbackground='#3E4149')

            button.grid(row=row, column=column,
                        padx=GRID_GAP[1], pady=GRID_GAP[0], sticky="nw")

            return button

        return make


UI.get_label = staticmethod(UI.get_label)
UI.get_button = staticmethod(UI.get_button)
