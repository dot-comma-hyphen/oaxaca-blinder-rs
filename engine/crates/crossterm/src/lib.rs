use std::io;

pub mod terminal {
    use std::io;
    pub fn enable_raw_mode() -> io::Result<()> { Ok(()) }
    pub fn disable_raw_mode() -> io::Result<()> { Ok(()) }
    pub fn size() -> io::Result<(u16, u16)> { Ok((80, 24)) }
    pub fn window_size() -> io::Result<WindowSize> { 
        Ok(WindowSize { rows: 24, columns: 80, width: 800, height: 600 }) 
    }
    pub fn is_raw_mode_enabled() -> io::Result<bool> { Ok(false) }

    pub struct WindowSize {
        pub rows: u16,
        pub columns: u16,
        pub width: u16,
        pub height: u16,
    }
}

pub mod style {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Color {
        Reset,
        Black,
        DarkGrey,
        Red,
        DarkRed,
        Green,
        DarkGreen,
        Yellow,
        DarkYellow,
        Blue,
        DarkBlue,
        Magenta,
        DarkMagenta,
        Cyan,
        DarkCyan,
        White,
        Grey,
        Rgb { r: u8, g: u8, b: u8 },
        AnsiValue(u8),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Attribute {
        Reset,
        Bold,
        Dim,
        Italic,
        Underlined,
        DoubleUnderlined,
        Underdashed,
        Underdotted,
        Undercurled,
        SlowBlink,
        RapidBlink,
        Reverse,
        Hidden,
        CrossedOut,
        Fraktur,
        NoBold,
        NormalIntensity,
        NoItalic,
        NoUnderline,
        NoBlink,
        NoReverse,
        NoHidden,
        NotCrossedOut,
        Framed,
        Encircled,
        OverLined,
        NoFramedEncircled,
        NotFramedOrEncircled,
        NoOverLined,
        NotOverLined,
    }
    
    pub struct ContentStyle {
        pub foreground_color: Option<Color>,
        pub background_color: Option<Color>,
        pub attributes: Vec<Attribute>,
    }
    
    impl ContentStyle {
        pub fn new() -> Self {
            Self { foreground_color: None, background_color: None, attributes: Vec::new() }
        }
    }
    
    pub trait Stylize: Sized {
        fn stylize(self) -> StyledContent<Self> {
            StyledContent::new(ContentStyle::new(), self)
        }
        fn with(self, color: Color) -> StyledContent<Self> {
            let mut style = ContentStyle::new();
            style.foreground_color = Some(color);
            StyledContent::new(style, self)
        }
        fn on(self, color: Color) -> StyledContent<Self> {
            let mut style = ContentStyle::new();
            style.background_color = Some(color);
            StyledContent::new(style, self)
        }
        fn attribute(self, attr: Attribute) -> StyledContent<Self> {
            let mut style = ContentStyle::new();
            style.attributes.push(attr);
            StyledContent::new(style, self)
        }
    }
    
    impl Stylize for String {}
    impl Stylize for &str {}
    impl Stylize for char {}
    
    pub fn style<D>(content: D) -> StyledContent<D> {
        StyledContent::new(ContentStyle::new(), content)
    }

    pub struct StyledContent<D> {
        style: ContentStyle,
        content: D,
    }
    
    impl<D> StyledContent<D> {
        pub fn new(style: ContentStyle, content: D) -> Self {
            Self { style, content }
        }
        pub fn with(mut self, color: Color) -> Self {
            self.style.foreground_color = Some(color);
            self
        }
        pub fn on(mut self, color: Color) -> Self {
            self.style.background_color = Some(color);
            self
        }
        pub fn attribute(mut self, attr: Attribute) -> Self {
            self.style.attributes.push(attr);
            self
        }
    }
    
    impl<D: std::fmt::Display> std::fmt::Display for StyledContent<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.content)
        }
    }
}

pub mod cursor {
    // Add mocks if needed
}

pub mod event {
    // Add mocks if needed
}

pub mod tty {
    pub trait IsTty {
        fn is_tty(&self) -> bool;
    }

    impl<T: ?Sized> IsTty for T {
        fn is_tty(&self) -> bool { false }
    }
}
